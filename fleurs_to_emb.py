import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
)  # Added E2T for completeness if ever needed
from tqdm import tqdm

# It's good practice to ensure fairseq2 is initialized if Sonar relies on it.
try:
    import fairseq2

    fairseq2.setup_fairseq2()
except ImportError:
    print(
        "Warning: fairseq2 not found or setup failed. Sonar models might not work as expected."
    )


def get_device():
    """Picks the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # For Apple Silicon
    #     return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings from FLEURS dataset using Sonar models."
    )
    parser.add_argument(
        "fleurs_code",
        type=str,
        help="FLEURS language code (e.g., 'ca_es', 'en_us', 'es_419').",
    )
    parser.add_argument(
        "encoder_name",
        type=str,
        help="Sonar encoder model card name (e.g., 'sonar_speech_encoder_spa', 'text_sonar_basic_encoder').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to process (default: 'test').",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=os.getenv("DATA", "./data"),  # Default to $DATA or ./data
        help="Base directory to save embeddings (under 'fleurs/<fleurs_code>/<split>/<encoder_name>').",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,  # Auto-detect if None
        help="Device to use ('cuda', 'cpu', 'mps'). If None, auto-detects.",
    )
    # For text_sonar_basic_encoder, we might need tokenizer and source_lang
    parser.add_argument(
        "--text_tokenizer",
        type=str,
        default="text_sonar_basic_encoder",  # Often same as encoder for text models
        help="Tokenizer card name, primarily for text encoders.",
    )
    parser.add_argument(
        "--source_lang_text",
        type=str,
        default=None,  # Will try to infer from fleurs_code or require it
        help="Source language for text embeddings (e.g., 'cat_Latn', 'eng_Latn'). Required for text_sonar_basic_encoder.",
    )

    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")

    # Construct output directory path
    output_dir = (
        Path(args.output_base_dir)
        / "fleurs"
        / args.fleurs_code
        / args.split
        / args.encoder_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Embeddings will be saved to: {output_dir}")

    print(
        f"Loading FLEURS dataset: google/fleurs, code: {args.fleurs_code}, split: {args.split}"
    )
    try:
        dataset = load_dataset(
            "google/fleurs", args.fleurs_code, split=args.split, streaming=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(
            "Please ensure the dataset name and configuration are correct and you have internet access."
        )
        return

    # Initialize model based on encoder_name
    model = None
    is_text_encoder = "text_" in args.encoder_name.lower()  # Simple heuristic

    if is_text_encoder:
        if not args.source_lang_text:
            # Attempt to infer source_lang from fleurs_code (very basic)
            lang_part = args.fleurs_code.split("_")[0]
            script_part = "Latn"  # Common default for FLEURS
            args.source_lang_text = f"{lang_part}_{script_part}"
            print(
                f"Inferred --source_lang_text as '{args.source_lang_text}'. Please verify or provide explicitly if incorrect."
            )

        print(
            f"Initializing TextToEmbeddingModelPipeline for '{args.encoder_name}' with tokenizer '{args.text_tokenizer}'"
        )
        try:
            model = TextToEmbeddingModelPipeline(
                encoder=args.encoder_name, tokenizer=args.text_tokenizer, device=device
            )
        except Exception as e:
            print(f"Error initializing TextToEmbeddingModelPipeline: {e}")
            return
    else:  # Assume speech encoder
        print(f"Initializing SpeechToEmbeddingModelPipeline for '{args.encoder_name}'")
        try:
            model = SpeechToEmbeddingModelPipeline(
                encoder=args.encoder_name, device=device
            )
        except Exception as e:
            print(f"Error initializing SpeechToEmbeddingModelPipeline: {e}")
            return

    if model is None:
        print("Model could not be initialized. Exiting.")
        return

    print(f"Processing batches with batch_size: {args.batch_size}")
    # Wrap dataset.batch with tqdm for a progress bar on batches
    # Note: For streaming datasets, total number of items might not be known for tqdm
    # We can estimate if we know the dataset size, or just count batches.
    # fleurs dataset object from `streaming=True` doesn't have a `__len__` directly.
    # We can iterate and count, or if the total number of samples for a split is known, use that.
    # For now, tqdm will just show iterations.

    for batch_idx, batch in enumerate(
        tqdm(dataset.batch(batch_size=args.batch_size), desc="Processing batches")
    ):
        embeddings = None
        if is_text_encoder:
            try:
                embeddings = model.predict(
                    batch["transcription"], source_lang=args.source_lang_text
                ).cpu()
            except Exception as e:
                print(
                    f"Error during text embedding prediction for batch {batch_idx}: {e}"
                )
                continue  # Skip to next batch
        else:  # Speech encoder
            try:
                # Prepare audio data: list of 1D tensors, then unsqueeze for batch dim for model
                audio_tensors = [
                    torch.tensor(item["array"], dtype=torch.float32).unsqueeze(0)
                    for item in batch["audio"]
                ]
                embeddings = model.predict(audio_tensors, batch_size=4).cpu()
            except Exception as e:
                print(
                    f"Error during speech embedding prediction for batch {batch_idx}: {e}"
                )
                # print(f"Audio batch structure: {batch['audio']}")
                # if audio_tensors:
                #     print(f"First audio tensor shape prepared for model: {audio_tensors[0].shape}")
                continue  # Skip to next batch

        if embeddings is None:
            print(
                f"No embeddings generated for batch {batch_idx}. Skipping save for this batch."
            )
            continue

        utt_ids = batch["id"]  # This is a list of unique IDs for each utterance
        # wav_ids are typically derived if needed, for FLEURS 'id' should be sufficient as primary key
        wav_ids = [item["path"].split("/")[1].split(".")[0] for item in batch["audio"]]
        # If 'path' exists and you need part of it:
        # wav_ids = [Path(item['path']).stem for item in batch['audio']] if 'path' in batch['audio'][0] else utt_ids
        # For fleurs, 'id' is usually the most reliable unique identifier for an utterance.

        for i, emb in enumerate(embeddings):
            # Use the utterance ID from the batch directly.
            # The 'id' field in FLEURS is typically a numeric string or int.
            # Ensure it's a string for filename.
            current_utt_id = f"{utt_ids[i]}_{wav_ids[i]}"
            try:
                torch.save(emb, output_dir / f"{current_utt_id}.pt")
            except Exception as e:
                print(f"Error saving embedding for utt_id {current_utt_id}: {e}")

    print("Embedding generation complete.")
    print(f"Embeddings saved in: {output_dir}")


if __name__ == "__main__":
    main()
