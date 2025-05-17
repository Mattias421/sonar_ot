import os
import sys
from pathlib import Path

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils import hpopt as hp
from speechbrain.utils.logger import get_logger
from torchdyn.core import NeuralODE

from prepare_json import prepare_json

logger = get_logger(__name__)

# Attempt to get DATA environment variable, default to current directory if not set
DATA_ENV_VAR = os.getenv("DATA")
if DATA_ENV_VAR is None:
    print(
        "Warning: $DATA environment variable not set. Using current directory '.' as base for data."
    )
    BASE_DATA_PATH = Path(".")
else:
    BASE_DATA_PATH = Path(DATA_ENV_VAR)


# Brain class for speech recognition training
class ASR(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        x0_speech = batch["speech_src"].data
        x1_text = batch["text"].data

        t, xt, ut = self.hparams.cfm.sample_location_and_conditional_flow(
            x0_speech, x1_text
        )

        vt = self.modules.model(t, xt)

        return vt, ut

    def compute_objectives(self, predictions, batch, stage):
        vt, ut = predictions
        loss = torch.mean((vt - ut) ** 2)

        if stage != sb.Stage.TRAIN:
            x0_speech = batch["speech_src"].data.to(self.device)
            node = NeuralODE(
                self.modules.model,
                solver=self.hparams.cfm_ode_solver,
                sensitivity="adjoint",
                atol=1e-3,
                rtol=1e-3,
            )

            with torch.no_grad():
                traj = node.trajectory(
                    x0_speech, t_span=torch.linspace(0, 1, self.hparams.cfm_n_ode_steps)
                )

            x1_text_pred = traj[-1]
            x1_text_ref = batch["text"].data.to(self.device)

            self.hparams.cosim.append(ids=batch["id"], x1=x1_text_pred, x2=x1_text_ref)

            self.hparams.l1_loss.append(
                ids=batch["id"],
                predictions=x1_text_pred,
                targets=batch["text"].data.to(self.device),
                reduction="batch",
            )

            if stage == sb.Stage.TEST:
                pred_outputs = Path(self.hparams.pred_outputs)
                pred_outputs.mkdir(exist_ok=True)

                for ID, pred in zip(batch["id"], x1_text_pred):
                    torch.save(pred, pred_outputs / f"{ID}.pt")

        return loss

    def on_stage_start(self, stage, epoch):
        self.epoch = epoch if (epoch is not None) else 0

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["cosim"] = self.hparams.cosim.summarize("average")
            stage_stats["l1_loss"] = self.hparams.l1_loss.summarize("average")

            self.hparams.cosim.clear()
            self.hparams.l1_loss.clear()

            stage_stats["cosdist"] = 1 - stage_stats["cosim"]

            self.hparams.epoch_counter.update_metric(
                stage_stats[self.hparams.validate_optim_metric]
            )
            hp.report_result(stage_stats)

            # Save the current checkpoint and delete previous checkpoints.
            if self.hparams.ckpt_enable:
                self.checkpointer.save_and_keep_only(
                    meta={"loss": stage_stats["loss"]},
                    min_keys=["loss"],
                )

            if stage == sb.Stage.TEST:
                save_folder = Path(self.hparams.output_folder) / "test_stats"
                save_folder.mkdir(parents=True, exist_ok=True)

                with open(save_folder / "cosim.txt", "w") as f:
                    self.hparams.cosim.write_stats(f)

                self.hparams.cosim.clear()

                with open(save_folder / "l1_loss.txt", "w") as f:
                    self.hparams.l1_loss.write_stats(f)

                self.hparams.l1_loss.clear()

            else:
                # The train_logger writes a summary to stdout and to the logfile.
                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch},
                    train_stats=self.train_stats,
                    valid_stats=stage_stats,
                )


def dataio_prepare(hparams, data_folder):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys that correspond
        to the DynamicItemDataset objects.
    """

    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes(hparams["encoder"], "text_sonar_basic_encoder")
    @sb.utils.data_pipeline.provides("speech_src", "text")
    def pipeline(src, txt):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sonar_speech_src = torch.load(src)
        sonar_text = torch.load(txt)
        return sonar_speech_src, sonar_text

    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            dynamic_items=[pipeline],
            replacements={"data_root": data_folder},
            output_keys=[
                "id",
                "speech_src",
                "text",
            ],
        )
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False

    return datasets


if __name__ == "__main__":
    # Reading command line arguments
    with hp.hyperparameter_optimization(
        objective_key="cosdist"
    ) as hp_ctx:  # <-- Initialize the context
        hparams_file, run_opts, overrides = hp_ctx.parse_arguments(
            sys.argv[1:]
        )  # <-- Replace sb with hp_ctx

        # Initialize ddp (useful only for multi-GPU DDP training)
        sb.utils.distributed.ddp_init_group(run_opts)

        # Load hyperparameters file with command-line overrides
        with open(hparams_file, encoding="utf-8") as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )

        data_folder = BASE_DATA_PATH / "fleurs" / hparams["lang_id_fleurs"]

        if hparams["scramble_train"]:
            scramble_config = {"embedding_type":"text_sonar_basic_encoder", "seed":hparams["seed"]}
        else:
            scramble_config = None

        # Data preparation, to be run on only one process.
        if not hparams["skip_prep"]:
            sb.utils.distributed.run_on_main(
                prepare_json,
                kwargs={
                    "data_root_folder": data_folder,
                    "save_json_train": hparams["train_annotation"],
                    "save_json_valid": hparams["valid_annotation"],
                    "save_json_test": hparams["test_annotation"],
                    "scramble_train_config": scramble_config
                },
            )
        # We can now directly create the datasets for training, valid, and test
        datasets = dataio_prepare(hparams, data_folder)

        # Trainer initialization
        asr_brain = ASR(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )

        # The `fit()` method iterates the training loop, calling the methods
        # necessary to update the parameters of the model. Since all objects
        # with changing state are managed by the Checkpointer, training can be
        # stopped at any point, and will be resumed on next call.
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            datasets["train"],
            datasets["valid"],
            train_loader_kwargs=hparams["train_dataloader_opts"],
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )

        if asr_brain.hparams.evaluate:
            test_stats = asr_brain.evaluate(
                test_set=datasets["test"],
                test_loader_kwargs=hparams["test_dataloader_opts"],
                min_key=hparams["validate_optim_metric"],
            )

        # The `fit()` method iterates the training loop, calling the methods
        # necessary to update the parameters of the model. Since all objects
        # with changing state are managed by the Checkpointer, training can be
        # stopped at any point, and will be resumed on next call.

        # test_stats = asr_brain.transcribe_sonar_speech(datasets["test"])
