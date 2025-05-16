import os
from pathlib import Path

import pandas as pd
import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from tqdm import tqdm

# --- Configuration ---
MODEL_ENCODER = "text_sonar_basic_encoder"
MODEL_TOKENIZER = (
    "text_sonar_basic_encoder"  # Often the same as encoder for Sonar text models
)
SOURCE_LANG = "bem_Latn"  # Bemba, Latin script

# Attempt to get DATA environment variable, default to current directory if not set
DATA_ENV_VAR = os.getenv("DATA")
if DATA_ENV_VAR is None:
    print(
        "Warning: $DATA environment variable not set. Using current directory '.' as base for data."
    )
    BASE_DATA_PATH = Path(".")
else:
    BASE_DATA_PATH = Path(DATA_ENV_VAR)

INPUT_TSV_DIR = BASE_DATA_PATH / "BembaSpeech" / "bem"
OUTPUT_TEXT_EMBEDDING_DIR = BASE_DATA_PATH / "BembaSpeech" / "bem" / "sonar_text"
TSV_FILES_TO_PROCESS = ["train.tsv", "dev.tsv", "test.tsv"]

# Columns in the TSV file (adjust if different)
AUDIO_COLUMN_NAME = "audio"  # or "path", "filename" etc.
SENTENCE_COLUMN_NAME = "sentence"  # or "transcription", "text" etc.


def ensure_dir_exists(path: Path):
    """Creates a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Input TSV directory: {INPUT_TSV_DIR.resolve()}")
    print(f"Output text embedding directory: {OUTPUT_TEXT_EMBEDDING_DIR.resolve()}")

    if not INPUT_TSV_DIR.is_dir():
        print(f"Error: Input TSV directory not found: {INPUT_TSV_DIR}")
        return

    ensure_dir_exists(OUTPUT_TEXT_EMBEDDING_DIR)

    # Determine device
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Initialize the TextToEmbeddingModel
    print(
        f"Loading Sonar text model: encoder='{MODEL_ENCODER}', tokenizer='{MODEL_TOKENIZER}'..."
    )
    try:
        t2vec_model = TextToEmbeddingModelPipeline(
            encoder=MODEL_ENCODER, tokenizer=MODEL_TOKENIZER, device=device
        )
    except Exception as e:
        print(f"Error initializing TextToEmbeddingModelPipeline: {e}")
        print(
            "Please ensure 'sonar-speech' and its dependencies are correctly installed."
        )
        return

    for tsv_filename in TSV_FILES_TO_PROCESS:
        tsv_path = INPUT_TSV_DIR / tsv_filename
        if not tsv_path.is_file():
            print(f"Warning: TSV file not found: {tsv_path}, skipping.")
            continue

        print(f"\nProcessing {tsv_filename}...")
        try:
            df = pd.read_csv(
                tsv_path, sep="\t", quoting=3
            )  # quoting=3 to handle potential quotes in sentences
        except Exception as e:
            print(f"Error reading TSV file {tsv_path}: {e}")
            continue

        if AUDIO_COLUMN_NAME not in df.columns:
            print(
                f"Error: Audio column '{AUDIO_COLUMN_NAME}' not found in {tsv_filename}. Available columns: {df.columns.tolist()}"
            )
            continue
        if SENTENCE_COLUMN_NAME not in df.columns:
            print(
                f"Error: Sentence column '{SENTENCE_COLUMN_NAME}' not found in {tsv_filename}. Available columns: {df.columns.tolist()}"
            )
            continue

        # Extract data for batch processing
        sentences_list = df[SENTENCE_COLUMN_NAME].astype(str).tolist()
        # The audio column might contain full paths like "audio/train/bem_0001.wav"
        # We need the stem of the filename, e.g., "bem_0001"
        audio_file_stems = [
            Path(audio_path).stem for audio_path in df[AUDIO_COLUMN_NAME].tolist()
        ]

        if not sentences_list:
            print(f"No sentences found in {tsv_filename}.")
            continue

        print(
            f"Computing embeddings for {len(sentences_list)} sentences from {tsv_filename}..."
        )
        try:
            # Predict embeddings in batch
            # The model handles tokenization and moving data to its device internally
            all_embeddings = t2vec_model.predict(
                sentences_list, source_lang=SOURCE_LANG
            )
            # all_embeddings shape will be [num_sentences, embedding_dim] (e.g., [N, 1024])
        except Exception as e:
            print(f"Error during embedding prediction for {tsv_filename}: {e}")
            continue

        print(f"Saving embeddings for {tsv_filename}...")
        for i, audio_stem in enumerate(
            tqdm(audio_file_stems, desc=f"Saving {tsv_filename}")
        ):
            embedding_tensor = all_embeddings[
                i
            ]  # This is already the [embedding_dim] tensor

            output_filename = f"{audio_stem}.pt"
            output_path = OUTPUT_TEXT_EMBEDDING_DIR / output_filename

            try:
                torch.save(
                    embedding_tensor.cpu(), output_path
                )  # Save tensor to CPU before disk
            except Exception as e:
                tqdm.write(
                    f"Error saving embedding for {audio_stem} (from {tsv_filename}): {e}"
                )

    print("\nProcessing complete.")
    print(f"Text embeddings saved in: {OUTPUT_TEXT_EMBEDDING_DIR.resolve()}")


if __name__ == "__main__":
    main()
