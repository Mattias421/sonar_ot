import json
import os
import random  # For scrambling
from pathlib import Path

import torch  # To optionally get embedding shapes

# from speechbrain.dataio.dataio import read_audio # Not needed
# from speechbrain.utils.data_utils import download_file, get_all_files # Partly replaced
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)
# SAMPLERATE = 16000 # Not relevant for embeddings


def prepare_json(
    data_root_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    splits_to_process=None,
    expected_embedding_types=None,  # e.g., ["speech", "text"] or None to auto-discover
    store_embedding_shapes=False,  # Option to load .pt and store shape
    scramble_train_config=None,  # dict: {"embedding_type": "text_embs", "seed": 42} or None
):
    """
    Prepares json files for a custom dataset based on pre-computed embeddings.
    The dataset is expected to be structured as:
    data_root_folder/SPLIT_NAME/EMBEDDING_TYPE_NAME/unique_id.pt

    Arguments
    ---------
    data_root_folder : str
        Path to the root folder of the dataset.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    splits_to_process : list of str, optional
        A list of split names to process (e.g., ["train", "validation", "test"]).
        Defaults to ["train", "validation", "test"].
    expected_embedding_types : list of str, optional
        A list of expected embedding type subfolder names (e.g., ["speech_emb", "text_emb"]).
        If None, it will try to auto-discover from the first split.
    store_embedding_shapes : bool
        If True, will load each .pt file to get its tensor shape and store it.
        This can be slow if there are many files or large embeddings.
    scramble_train_config : dict, optional
        Configuration for scrambling a specific embedding type in the train set.
        Example: {"embedding_type": "text_sonar_basic_encoder", "seed": 42}.
        If "embedding_type" is provided, paths for this type in the training set
        will be scrambled. "seed" is optional for reproducible scrambling.
        If None, no scrambling is performed.

    Returns
    -------
    None
    """
    data_root = Path(data_root_folder)
    if not data_root.is_dir():
        logger.error(f"Data root folder {data_root} not found.")
        return

    # Check if this phase is already done (if so, skip it)
    # If scramble_train_config is active, we might need to regenerate even if files exist.
    # The current skip logic handles this: it skips only if scramble_train_config is None.
    if (
        skip(save_json_train, save_json_valid, save_json_test)
        and not scramble_train_config
    ):
        logger.info(
            "Preparation completed in previous run (and no scrambling requested), skipping."
        )
        return
    elif (
        skip(save_json_train, save_json_valid, save_json_test) and scramble_train_config
    ):
        logger.info(
            "JSON files exist, but re-generating because scramble_train_config is active for the train split."
        )
        # We might only need to regenerate train, but for simplicity, this implies all are checked.
        # The create_json_for_split_embeddings will only apply scramble to train.

    if splits_to_process is None:
        splits_to_process = ["train", "validation", "test"]

    split_to_json_map = {
        "train": Path(save_json_train),
        "validation": Path(save_json_valid),
        "test": Path(save_json_test),
    }

    # Auto-discover embedding types from the first available split if not provided
    # Use a local copy for current_expected_embedding_types
    current_expected_embedding_types = (
        list(expected_embedding_types) if expected_embedding_types else None
    )
    if current_expected_embedding_types is None:
        logger.info("Attempting to auto-discover embedding types...")
        discovered_types = False
        for split_name_discover in splits_to_process:
            split_path_discover = data_root / split_name_discover
            if split_path_discover.is_dir():
                current_expected_embedding_types = sorted(
                    [d.name for d in split_path_discover.iterdir() if d.is_dir()]
                )
                if current_expected_embedding_types:
                    logger.info(
                        f"Auto-discovered embedding types: {current_expected_embedding_types} from split '{split_name_discover}'"
                    )
                    discovered_types = True
                    break
        if not discovered_types or not current_expected_embedding_types:
            logger.error(
                "Could not auto-discover embedding types. Please provide them via 'expected_embedding_types'."
            )
            return
    else:
        logger.info(
            f"Using provided embedding types: {current_expected_embedding_types}"
        )

    for split_name in splits_to_process:
        logger.info(f"Processing split: {split_name}")
        split_path = data_root / split_name
        output_json_file = split_to_json_map.get(split_name)

        if output_json_file is None:
            logger.warning(
                f"No output JSON path defined for split '{split_name}', skipping."
            )
            continue

        # Ensure output directory exists
        output_json_file.parent.mkdir(parents=True, exist_ok=True)

        if not split_path.is_dir():
            logger.warning(
                f"Split directory {split_path} not found, skipping split '{split_name}'."
            )
            if not output_json_file.exists():  # Create an empty JSON if the directory doesn't exist but a save path is specified
                with open(output_json_file, mode="w", encoding="utf-8") as json_f:
                    json.dump({}, json_f, indent=2)
                logger.info(f"Created empty JSON for missing split: {output_json_file}")
            continue

        scramble_config_for_split = None
        if (
            split_name == "train"
            and scramble_train_config
            and "embedding_type" in scramble_train_config
        ):
            scramble_embedding_type_to_check = scramble_train_config["embedding_type"]
            if scramble_embedding_type_to_check not in current_expected_embedding_types:
                logger.warning(
                    f"Scramble embedding type '{scramble_embedding_type_to_check}' "
                    f"is not in the list of expected_embedding_types: {current_expected_embedding_types}. "
                    "Scrambling will be skipped for this type."
                )
            else:
                scramble_config_for_split = scramble_train_config

        create_json_for_split_embeddings(
            data_root_path=data_root,
            split_name=split_name,
            embedding_types=current_expected_embedding_types,
            json_file_path=output_json_file,
            store_embedding_shapes=store_embedding_shapes,
            scramble_config=scramble_config_for_split,
        )


def create_json_for_split_embeddings(
    data_root_path: Path,
    split_name: str,
    embedding_types: list,
    json_file_path: Path,
    store_embedding_shapes: bool,
    scramble_config: dict = None,  # e.g. {"embedding_type": "text_embs", "seed": 42}
):
    """
    Creates a JSON file for a specific split by finding matching .pt embedding files.

    Arguments
    ---------
    data_root_path : Path
        Path to the root folder of the dataset.
    split_name : str
        Name of the current split (e.g., "train").
    embedding_types : list of str
        List of subfolder names representing different embedding types.
    json_file_path : Path
        The path of the output json file for this split.
    store_embedding_shapes : bool
        If True, load .pt files to store tensor shapes.
    scramble_config : dict, optional
        Configuration for scrambling a specific embedding type.
        Applies if this function is called for the "train" split and this config is provided.
        Example: {"embedding_type": "text_sonar_basic_encoder", "seed": 42}.
    """

    json_dict = {}
    split_root_path = data_root_path / split_name

    if not embedding_types:
        logger.warning(
            f"No embedding types defined for split {split_name}. Cannot proceed to create JSON."
        )
        if not json_file_path.exists():
            with open(json_file_path, mode="w", encoding="utf-8") as json_f:
                json.dump({}, json_f, indent=2)
            logger.info(
                f"Created empty JSON for split with no embedding types: {json_file_path}"
            )
        return

    primary_emb_type_path = split_root_path / embedding_types[0]
    if not primary_emb_type_path.is_dir():
        logger.warning(
            f"Primary embedding type directory '{primary_emb_type_path}' for IDs discovery "
            f"not found in split '{split_name}'. Creating empty JSON."
        )
        if not json_file_path.exists():
            with open(json_file_path, mode="w", encoding="utf-8") as json_f:
                json.dump({}, json_f, indent=2)
        return

    unique_ids = sorted(
        [pt_file.stem for pt_file in primary_emb_type_path.glob("*.pt")]
    )
    logger.info(
        f"Found {len(unique_ids)} unique IDs in '{primary_emb_type_path}' for split '{split_name}'."
    )

    for item_id in unique_ids:
        entry = {}
        all_types_present_for_id = True

        for emb_type in embedding_types:
            emb_file_path = split_root_path / emb_type / f"{item_id}.pt"

            if emb_file_path.is_file():
                relative_emb_path = emb_file_path.relative_to(data_root_path)
                entry[f"{emb_type}"] = os.path.join(
                    "{data_root}", str(relative_emb_path)
                )

                if store_embedding_shapes:
                    try:
                        tensor = torch.load(emb_file_path, map_location="cpu")
                        entry[f"{emb_type}_shape"] = list(tensor.shape)
                    except Exception as e:
                        logger.warning(
                            f"Could not load or get shape for {emb_file_path}: {e}"
                        )
                        entry[f"{emb_type}_shape"] = None
            else:
                logger.debug(
                    f"Embedding file not found for id '{item_id}', type '{emb_type}': {emb_file_path}"
                )
                all_types_present_for_id = False
                break

        if all_types_present_for_id and entry:
            json_dict[item_id] = entry

    # Scrambling logic, applied only if split_name is "train" and scramble_config is valid
    if split_name == "train" and scramble_config:
        scramble_embedding_type = scramble_config.get("embedding_type")
        scramble_seed = scramble_config.get("seed")  # Can be None

        # Scramble_embedding_type should already be validated to be in embedding_types by prepare_json
        # but we check again for safety / direct calls.
        if scramble_embedding_type and scramble_embedding_type in embedding_types:
            logger.info(
                f"Attempting to scramble '{scramble_embedding_type}' paths for train split (seed: {scramble_seed})."
            )

            if scramble_seed is not None:
                random.seed(scramble_seed)

            item_ids_for_scrambling = []
            original_paths_to_scramble = []

            # Collect item_ids and paths for the target type from valid entries in json_dict
            for item_id, data_entry in json_dict.items():
                if (
                    scramble_embedding_type in data_entry
                ):  # data_entry is guaranteed to have all embedding_types
                    original_paths_to_scramble.append(
                        data_entry[scramble_embedding_type]
                    )
                    item_ids_for_scrambling.append(item_id)

            if not original_paths_to_scramble:
                logger.warning(
                    f"No items with '{scramble_embedding_type}' found in train split entries to scramble."
                )
            elif len(original_paths_to_scramble) <= 1:
                logger.warning(
                    f"Only {len(original_paths_to_scramble)} item(s) with '{scramble_embedding_type}' "
                    "in train split. Scrambling will have no (or trivial) effect."
                )
            else:
                shuffled_paths = list(original_paths_to_scramble)

                # Try to ensure paths are different from original positions (minimize self-assignments)
                # This is a heuristic for derangement.
                max_attempts = 20  # More attempts for better chance of derangement
                best_shuffled_paths = list(
                    shuffled_paths
                )  # Store the shuffle with fewest self-assignments
                min_self_assigned = len(
                    shuffled_paths
                )  # Initialize with max possible self-assignments

                for attempt in range(max_attempts):
                    random.shuffle(shuffled_paths)
                    num_self_assigned = 0
                    for i in range(len(original_paths_to_scramble)):
                        if original_paths_to_scramble[i] == shuffled_paths[i]:
                            num_self_assigned += 1

                    if num_self_assigned < min_self_assigned:
                        min_self_assigned = num_self_assigned
                        best_shuffled_paths = list(
                            shuffled_paths
                        )  # Found a better shuffle

                    if min_self_assigned == 0:  # Perfect derangement achieved
                        break

                shuffled_paths = best_shuffled_paths  # Use the best shuffle found

                if min_self_assigned > 0:
                    logger.warning(
                        f"Scrambling for '{scramble_embedding_type}': {min_self_assigned} item(s) "
                        f"out of {len(original_paths_to_scramble)} might still point to their original embedding "
                        f"for this type after {max_attempts} shuffling attempts."
                    )
                else:
                    logger.info(
                        f"Successfully scrambled '{scramble_embedding_type}' paths for {len(original_paths_to_scramble)} items "
                        "with no self-assignments."
                    )

                # Assign the shuffled paths back. item_ids_for_scrambling order matches original_paths_to_scramble order.
                for i, item_id in enumerate(item_ids_for_scrambling):
                    json_dict[item_id][scramble_embedding_type] = shuffled_paths[i]
                    # Note: The _shape field for the scrambled type (if store_embedding_shapes is True)
                    # will still correspond to its *original* file, as shapes are computed before this scrambling step.
                    # This is an important consideration if the shapes of the scrambled files are different and relevant.
        elif scramble_embedding_type:
            logger.warning(
                f"Scramble type '{scramble_embedding_type}' not in {embedding_types}, skipping scramble."
            )

    # Writing the dictionary to the json file
    json_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file_path, mode="w", encoding="utf-8") as json_f:
        json.dump(json_dict, json_f, indent=2, sort_keys=True)

    logger.info(f"{json_file_path} successfully created with {len(json_dict)} entries.")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.
    """
    for filename in filenames:
        if not Path(filename).is_file():
            return False
    return True
