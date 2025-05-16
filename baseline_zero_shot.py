#!/usr/bin/env/python3
"""Recipe for training a sequence-to-sequence ASR system with mini-librispeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beam search coupled with a neural
language model.

To run this recipe, do the following:
> python train.py train.yaml

With the default hyperparameters, the system employs an LSTM encoder.
The decoder is based on a standard  GRU. Beam search coupled with an RNN language
model is used on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the mini-librispeech
dataset. Note that this is a tiny dataset used here just to
provide a working example. To achieve a better performance you have to train with
larger datasets, such as the full LibriSpeech one. In this case, to allow the
model to converge, we pre-train it with a bigger one (trained on the full librispeech
with the seq2seq 1k BPE recipe).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE).

This recipe assumes that the tokenizer and the LM are already trained.
To avoid token mismatches, the tokenizer used for the acoustic model is
the same use for the LM.  The recipe downloads the pre-trained tokenizer
and LM.

If you would like to train a full system from scratch do the following:
1- Train a tokenizer (see ../Tokenizer)
2- Train a language model (see ../LM)
3- Train the speech recognizer (with this code).


Authors
 * Mirco Ravanelli 2020
 * Ju-Chieh Chou 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
"""

import os
import sys
from pathlib import Path

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
from speechbrain.utils.logger import get_logger
from tqdm import tqdm

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

    def transcribe_sonar_speech(self, dataset, split):
        decoder = EmbeddingToTextModelPipeline(
            decoder=self.hparams.model,
            tokenizer=self.hparams.tokenizer,
            device=torch.device(self.device),
        )

        self.hparams.wer_src.clear()
        self.hparams.cer_src.clear()
        self.hparams.wer_text.clear()
        self.hparams.cer_text.clear()
        self.hparams.cosim_src.clear()
        self.hparams.cosim_tgt.clear()
        self.hparams.bertscore_src.clear()
        self.hparams.bleu_src.clear()
        self.hparams.bertscore_tgt.clear()
        self.hparams.bleu_tgt.clear()

        for item in tqdm(dataset):
            speech_src = item["speech_src"].to(self.device)[None]

            text_hyp = decoder.predict(speech_src, target_lang=self.hparams.target_lang)
            text_ref = [item["transcription_cat"]]

            self.hparams.bertscore_src.append(
                ids=[item["id"]],
                predict=text_hyp,
                target=text_ref,
            )
            self.hparams.bleu_src.append(
                ids=[item["id"]],
                predict=text_hyp,
                targets=[text_ref],
            )

            text_hyp = [t.split(" ") for t in text_hyp]

            self.hparams.wer_src.append(
                ids=[item["id"]],
                predict=text_hyp,
                target=text_ref,
            )
            self.hparams.cer_src.append(
                ids=[item["id"]],
                predict=text_hyp,
                target=text_ref,
            )

            speech_tgt = item["speech_tgt"].to(self.device)[None]

            text_hyp = decoder.predict(speech_tgt, target_lang=self.hparams.target_lang)

            self.hparams.bertscore_tgt.append(
                ids=[item["id"]],
                predict=text_hyp,
                target=text_ref,
            )
            self.hparams.bleu_tgt.append(
                ids=[item["id"]],
                predict=text_hyp,
                targets=[text_ref],
            )

            text_hyp = [t.split(" ") for t in text_hyp]

            self.hparams.wer_tgt.append(
                ids=[item["id"]],
                predict=text_hyp,
                target=text_ref,
            )
            self.hparams.cer_tgt.append(
                ids=[item["id"]],
                predict=text_hyp,
                target=text_ref,
            )

            text = item["text"].to(self.device)[None]

            text_hyp = decoder.predict(text, target_lang=self.hparams.target_lang)
            text_hyp = [t.split(" ") for t in text_hyp]

            self.hparams.wer_text.append(
                ids=[item["id"]],
                predict=text_hyp,
                target=text_ref,
            )
            self.hparams.cer_text.append(
                ids=[item["id"]],
                predict=text_hyp,
                target=text_ref,
            )

            self.hparams.cosim_src.append(
                ids=[item["id"]],
                x1=speech_src,
                x2=text,
            )

            self.hparams.cosim_tgt.append(
                ids=[item["id"]],
                x1=speech_tgt,
                x2=text,
            )

        output_folder = Path(self.hparams.output_folder)

        with open(output_folder / f"{split}_wer_speech_tgt.txt", "w") as f:
            self.hparams.wer_tgt.write_stats(f)

        with open(output_folder / f"{split}_wer_speech_tgt.txt", "w") as f:
            self.hparams.wer_tgt.write_stats(f)

        with open(output_folder / f"{split}_cer_speech_tgt.txt", "w") as f:
            self.hparams.cer_tgt.write_stats(f)

        with open(output_folder / f"{split}_wer_speech_src.txt", "w") as f:
            self.hparams.wer_src.write_stats(f)

        with open(output_folder / f"{split}_cer_speech_src.txt", "w") as f:
            self.hparams.cer_src.write_stats(f)

        with open(output_folder / f"{split}_wer_text.txt", "w") as f:
            self.hparams.wer_text.write_stats(f)

        with open(output_folder / f"{split}_cer_text.txt", "w") as f:
            self.hparams.cer_text.write_stats(f)

        with open(output_folder / f"{split}_cosim_tgt.txt", "w") as f:
            self.hparams.cosim_tgt.write_stats(f)

        with open(output_folder / f"{split}_cosim_src.txt", "w") as f:
            self.hparams.cosim_src.write_stats(f)

        with open(output_folder / f"{split}_bleu_src.txt", "w") as f:
            self.hparams.bleu_src.write_stats(f)

        with open(output_folder / f"{split}_bleu_tgt.txt", "w") as f:
            self.hparams.bleu_tgt.write_stats(f)

        with open(output_folder / f"{split}_bertscore_src.txt", "w") as f:
            self.hparams.bertscore_src.write_stats(f)

        with open(output_folder / f"{split}_bertscore_tgt.txt", "w") as f:
            self.hparams.bertscore_tgt.write_stats(f)


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
    tgt = "cat"
    src = "spa"

    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes(
        f"sonar_speech_encoder_{tgt}",
        f"sonar_speech_encoder_{src}",
        "text_sonar_basic_encoder",
    )
    @sb.utils.data_pipeline.provides("speech_src", "speech_tgt", "text")
    def pipeline(tgt, src, txt):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sonar_speech_tgt = torch.load(tgt)
        sonar_speech_src = torch.load(src)
        sonar_text = torch.load(txt)
        return sonar_speech_tgt, sonar_speech_src, sonar_text

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
                "speech_tgt",
                "text",
                "transcription_cat",
                "transcription_en",
            ],
        )
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False

    return datasets


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

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

    # Trainer initialization
    asr_brain = ASR(
        hparams=hparams,
        run_opts=run_opts,
    )

    data_folder = BASE_DATA_PATH / "fleurs" / hparams["lang_id_fleurs"]

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        sb.utils.distributed.run_on_main(
            prepare_json,
            kwargs={
                "data_root_folder": data_folder,
                "save_json_train": hparams["train_annotation"],
                "save_json_valid": hparams["valid_annotation"],
                "save_json_test": hparams["test_annotation"],
            },
        )
    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams, data_folder)

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.

    test_stats = asr_brain.transcribe_sonar_speech(datasets["test"], split="test")
    valid_stats = asr_brain.transcribe_sonar_speech(datasets["valid"], split="valid")
