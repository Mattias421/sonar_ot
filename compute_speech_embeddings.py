import os
from pathlib import Path

import torch
import torchaudio
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from tqdm import tqdm

# --- Configuration ---
TARGET_SAMPLE_RATE = 16000
MODEL_ENCODER = "sonar_speech_encoder_swh"
# Attempt to get DATA environment variable, default to current directory if not set
DATA_ENV_VAR = os.getenv("DATA")
if DATA_ENV_VAR is None:
    print(
        "Warning: $DATA environment variable not set. Using current directory '.' as base for data."
    )
    BASE_DATA_PATH = Path(".")
else:
    BASE_DATA_PATH = Path(DATA_ENV_VAR)

INPUT_AUDIO_DIR = BASE_DATA_PATH / "BembaSpeech" / "bem" / "audio"
OUTPUT_EMBEDDING_DIR = BASE_DATA_PATH / "BembaSpeech" / "bem" / "sonar_speech"
SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")  # Add more if needed


def ensure_dir_exists(path: Path):
    """Creates a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Input audio directory: {INPUT_AUDIO_DIR.resolve()}")
    print(f"Output embedding directory: {OUTPUT_EMBEDDING_DIR.resolve()}")

    if not INPUT_AUDIO_DIR.is_dir():
        print(f"Error: Input audio directory not found: {INPUT_AUDIO_DIR}")
        return

    ensure_dir_exists(OUTPUT_EMBEDDING_DIR)

    # Initialize the SpeechToEmbeddingModel
    print(f"Loading Sonar model: {MODEL_ENCODER}...")
    try:
        s2vec_model = SpeechToEmbeddingModelPipeline(
            encoder=MODEL_ENCODER, device=torch.device("cuda:0")
        )
        # You can optionally move the model to GPU if available
        if torch.cuda.is_available():
            print("CUDA is available. Moving model to GPU.")
            # s2vec_model.cuda()
        else:
            print("CUDA not available. Using CPU.")

    except Exception as e:
        print(f"Error initializing SpeechToEmbeddingModelPipeline: {e}")
        print(
            "Please ensure 'sonar-speech' and its dependencies (like 'torch') are correctly installed."
        )
        return

    audio_files = []
    for ext in SUPPORTED_EXTENSIONS:
        audio_files.extend(list(INPUT_AUDIO_DIR.glob(f"*{ext}")))

    if not audio_files:
        print(
            f"No audio files found in {INPUT_AUDIO_DIR} with extensions {SUPPORTED_EXTENSIONS}"
        )
        return

    print(f"Found {len(audio_files)} audio files to process.")

    for audio_file_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Load audio file
            waveform, sr = torchaudio.load(audio_file_path)
            waveform = waveform.cuda()

            # Resample if necessary
            if sr != TARGET_SAMPLE_RATE:
                # print(f"Resampling {audio_file_path.name} from {sr}Hz to {TARGET_SAMPLE_RATE}Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=TARGET_SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Sonar model expects a list of waveforms.
            # waveform from torchaudio.load is typically [channels, samples].
            # The example s2vec_model.predict([inp]) suggests it handles this.
            # If your model expects mono, you might need:
            # if waveform.shape[0] > 1:
            #     waveform = waveform.mean(dim=0, keepdim=True) # Average channels to mono
            # or waveform = waveform[0, :].unsqueeze(0) # Take first channel

            embedding = s2vec_model.predict([waveform])

            # embedding will be torch.Size([1, 1024]) since we process one file at a time
            # We want to save the actual embedding tensor of shape [1024]
            embedding_tensor = embedding.squeeze(0)  # Remove the batch dimension

            # Define output path
            output_filename = audio_file_path.stem + ".pt"  # e.g., audio_1.pt
            output_path = OUTPUT_EMBEDDING_DIR / output_filename

            # Save the embedding
            torch.save(embedding_tensor.cpu(), output_path)
            # tqdm.write(f"Saved embedding for {audio_file_path.name} to {output_path}")

        except Exception as e:
            tqdm.write(f"Error processing {audio_file_path.name}: {e}")

    print("Processing complete.")
    print(f"Embeddings saved in: {OUTPUT_EMBEDDING_DIR.resolve()}")


if __name__ == "__main__":
    main()
