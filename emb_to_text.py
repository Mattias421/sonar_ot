import argparse
from pathlib import Path
from typing import List, Optional, Union  # For type hinting

import fairseq2  # Required for setup by sonar/fairseq2
import pandas as pd  # For creating and sorting the TSV
import torch
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
from torch.cuda import is_available as is_cuda_available
from tqdm import tqdm


def load_embeddings_from_folder(
    folder_path: Path,
) -> tuple[list[torch.Tensor], list[str]]:
    """Loads all .pt files from a folder into a list of tensors and their stems."""
    embeddings = []
    file_stems = []
    file_paths = sorted(list(folder_path.glob("*.pt")))

    if not file_paths:
        print(f"Warning: No .pt files found in {folder_path}")
        return [], []

    print(f"Loading {len(file_paths)} embedding files from {folder_path}...")
    for pt_file in tqdm(file_paths, desc="Loading .pt files"):
        try:
            emb = torch.load(pt_file, map_location="cpu")  # Load to CPU
            embeddings.append(emb)
            file_stems.append(pt_file.stem)
        except Exception as e:
            print(f"Warning: Could not load {pt_file}: {e}")
    return embeddings, file_stems


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe embeddings from .pt files and save to hyps.tsv and hyps.txt."
    )
    parser.add_argument(
        "embedding_folder",
        type=str,
        help="Path to the folder containing .pt embedding files.",
    )
    parser.add_argument(
        "--output-dir",  # Changed to optional argument
        type=str,
        default=None,  # Default is None, will be set to embedding_folder if not provided
        help="Directory to save the output files (hyps.tsv and hyps.txt). Defaults to the embedding_folder.",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        required=True,
        help="Target language code for transcription (e.g., 'eng_Latn', 'fra_Latn').",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="text_sonar_basic_decoder",
        help="Sonar text decoder model card name.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="text_sonar_basic_decoder",
        help="Sonar text tokenizer card name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for the .predict() method (default: 32).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Maximum sequence length for generation (passed to generator_kwargs, default: 256).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for beam search decoding (default: 5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if is_cuda_available() else "cpu",
        help="Device to use ('cuda' or 'cpu', default: auto-detect).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Data type for the model (e.g., 'float16', 'bfloat16', 'float32'). If None, uses model default.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="hyps",
        help="Prefix for the output files (e.g., 'hyps' -> hyps.tsv, hyps.txt). Default: 'hyps'.",
    )

    args = parser.parse_args()

    embedding_folder_path = Path(args.embedding_folder)

    # Determine output directory
    if args.output_dir is None:
        output_dir_path = embedding_folder_path
    else:
        output_dir_path = Path(args.output_dir)

    output_dir_path.mkdir(
        parents=True, exist_ok=True
    )  # Create output directory if it doesn't exist
    print(f"Output files will be saved in: {output_dir_path}")

    output_tsv_file_path = output_dir_path / f"{args.output_prefix}.tsv"
    output_txt_file_path = output_dir_path / f"{args.output_prefix}.txt"

    if not embedding_folder_path.is_dir():
        print(f"Error: Embedding folder not found: {embedding_folder_path}")
        return

    torch_dtype_val: Optional[torch.dtype] = None
    if args.dtype:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype_val = dtype_map.get(args.dtype.lower())
        if torch_dtype_val is None:
            print(f"Warning: Unsupported dtype '{args.dtype}'. Using model default.")

    all_embeddings, file_stems = load_embeddings_from_folder(embedding_folder_path)
    if not all_embeddings:
        print("No embeddings loaded. Exiting.")
        return

    try:
        try:
            input_for_pipeline: Union[torch.Tensor, List[torch.Tensor]] = torch.stack(
                all_embeddings
            )
            # print(f"Stacked embeddings into a tensor of shape: {input_for_pipeline.shape}")
        except RuntimeError:
            # print("Could not stack all embeddings (likely due to varying shapes). Passing as a list of tensors.")
            input_for_pipeline = all_embeddings
    except Exception as e:
        print(f"Error preparing embeddings for pipeline: {e}")
        return

    print(
        f"Initializing model: Decoder='{args.decoder}', Tokenizer='{args.tokenizer}', Device='{args.device}'"
    )
    try:
        fairseq2.setup_fairseq2()
        vec2text_model = EmbeddingToTextModelPipeline(
            decoder=args.decoder,
            tokenizer=args.tokenizer,
            device=torch.device(args.device),
            dtype=torch_dtype_val,
        )
    except Exception as e:
        print(f"Failed to initialize EmbeddingToTextModelPipeline: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"Transcribing {len(all_embeddings)} embeddings...")
    try:
        reconstructed_texts = vec2text_model.predict(
            inputs=input_for_pipeline,
            target_lang=args.target_lang,
            batch_size=args.batch_size,
            progress_bar=True,
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback

        traceback.print_exc()
        return

    if len(reconstructed_texts) != len(file_stems):
        print(
            f"Warning: Mismatch! Number of transcriptions ({len(reconstructed_texts)}) "
            f"does not equal number of input files/stems ({len(file_stems)})."
        )

    results_data = []
    for i, text in enumerate(reconstructed_texts):
        utt_id = file_stems[i] if i < len(file_stems) else f"error_id_mismatch_{i}"
        results_data.append({"id": utt_id, "transcription": text})

    df_results = pd.DataFrame(results_data)
    try:
        df_results["id_numeric"] = pd.to_numeric(df_results["id"])
        df_results_sorted = df_results.sort_values(by="id_numeric").drop(
            columns=["id_numeric"]
        )
        # print("Sorted results by numeric interpretation of IDs.")
    except (ValueError, TypeError):
        df_results_sorted = df_results.sort_values(by="id")
        # print("Sorted results by lexicographical (string) order of IDs.")

    print(f"Saving sorted (ID, Transcription) to {output_tsv_file_path}...")
    df_results_sorted.to_csv(
        output_tsv_file_path,
        sep="\t",
        header=False,
        index=False,
        columns=["id", "transcription"],
    )

    print(f"Saving sorted transcriptions (only text) to {output_txt_file_path}...")
    with open(output_txt_file_path, "w", encoding="utf-8") as f_txt:
        for transcription in df_results_sorted["transcription"]:
            f_txt.write(transcription + "\n")

    print(f"Transcription complete. Outputs saved in {output_dir_path}")


if __name__ == "__main__":
    main()
