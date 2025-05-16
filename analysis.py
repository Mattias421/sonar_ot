import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchcfm import OTPlanSampler  # Assuming torchcfm is installed

# UMAP, TSNE, PCA are imported dynamically


def find_matching_files(speech_dir: Path, text_dir: Path):
    """Finds .pt files with common stems in speech and text directories."""
    if not speech_dir.is_dir():
        raise FileNotFoundError(f"Speech embedding directory not found: {speech_dir}")
    if not text_dir.is_dir():
        raise FileNotFoundError(f"Text embedding directory not found: {text_dir}")

    speech_files = {p.stem: p for p in speech_dir.glob("*.pt")}
    text_files = {p.stem: p for p in text_dir.glob("*.pt")}

    common_stems = sorted(list(speech_files.keys() & text_files.keys()))  # Intersection

    if not common_stems:
        print(
            f"Warning: No common .pt files found between {speech_dir} and {text_dir}."
        )
        return [], {}, {}

    print(
        f"Found {len(common_stems)} common embedding files (based on filename stems)."
    )
    return common_stems, speech_files, text_files


def l2_norm(speech_dir: Path, text_dir: Path):
    print(f"\n--- Calculating L2 Norm for embeddings in {speech_dir} vs {text_dir} ---")

    try:
        common_ids, speech_path_map, text_path_map = find_matching_files(
            speech_dir, text_dir
        )
    except FileNotFoundError as e:
        print(e)
        return

    if not common_ids:
        return

    total_norm = 0
    count = 0

    for file_id in common_ids:
        speech_emb_file = speech_path_map[file_id]
        text_emb_file = text_path_map[file_id]

        try:
            speech_emb = torch.load(speech_emb_file, map_location="cpu")
            text_emb = torch.load(text_emb_file, map_location="cpu")

            norm_diff = torch.linalg.norm(speech_emb - text_emb)
            print(f"L2 norm for {file_id}: {norm_diff.item():.4f}")
            total_norm += norm_diff.item()
            count += 1
        except Exception as e:
            print(
                f"Error processing file {file_id} ({speech_emb_file.name}/{text_emb_file.name}): {e}"
            )

    if count > 0:
        print(f"\nAverage L2 norm for {count} pairs: {total_norm / count:.4f}")
    else:
        print("No valid embedding pairs processed to compare.")


def visualize_embeddings(
    speech_dir: Path,
    text_dir: Path,
    reducer_type: str = "umap",
    N: int = 100,
    show_plots: bool = True,
    draw_lines: bool = True,
    use_ot_sampler: bool = False,
    output_dir: Path = None,  # Optional output directory
):
    print(f"\n--- Visualizing Embeddings using {reducer_type.upper()} (N={N}) ---")
    print(f"Speech dir: {speech_dir}")
    print(f"Text dir: {text_dir}")

    if output_dir is None:
        output_dir = speech_dir  # Default to saving in speech_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        common_ids_all, speech_path_map, text_path_map = find_matching_files(
            speech_dir, text_dir
        )
    except FileNotFoundError as e:
        print(e)
        return

    if not common_ids_all:
        return

    speech_data = []
    text_data = []
    processed_ids = []

    print(f"Attempting to load up to {N} common pairs for visualization...")
    # Shuffle if N is less than total, to get a random sample. Otherwise, take the first N.
    # For simplicity, we'll just take the first N common IDs as they are sorted.
    # If random sampling is desired, uncomment:
    # import random
    # if N < len(common_ids_all):
    #    ids_to_process = random.sample(common_ids_all, N)
    # else:
    #    ids_to_process = common_ids_all
    ids_to_process = common_ids_all[:N]

    for file_id in ids_to_process:
        # if len(processed_ids) >= N: # Already handled by slicing ids_to_process
        #     break

        speech_emb_file = speech_path_map[file_id]
        text_emb_file = text_path_map[file_id]

        # Existence is already confirmed by find_matching_files
        try:
            speech_emb = torch.load(speech_emb_file, map_location="cpu")
            text_emb = torch.load(text_emb_file, map_location="cpu")

            speech_data.append(speech_emb.numpy())
            text_data.append(text_emb.numpy())
            processed_ids.append(file_id)
        except Exception as e:
            print(
                f"Skipping {file_id} ({speech_emb_file.name}/{text_emb_file.name}) due to loading error: {e}"
            )

    num_samples_loaded = len(speech_data)
    if num_samples_loaded == 0:
        print("No valid embedding pairs loaded for visualization.")
        return

    print(f"Loaded {num_samples_loaded} pairs for visualization.")

    speech_data_np = np.array(speech_data)
    text_data_np = np.array(text_data)

    if reducer_type == "umap":
        from umap import UMAP

        reducer_instance = UMAP(n_components=2, random_state=42)
    elif reducer_type == "tsne":
        from sklearn.manifold import TSNE

        # Ensure perplexity is less than the number of samples
        perplexity_val = (
            min(30, num_samples_loaded - 1) if num_samples_loaded > 1 else 5
        )
        reducer_instance = TSNE(
            n_components=2, random_state=42, perplexity=perplexity_val
        )
    elif reducer_type == "pca":
        from sklearn.decomposition import PCA

        reducer_instance = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unsupported reducer type: {reducer_type}")

    all_data_np = np.concatenate((speech_data_np, text_data_np), axis=0)

    print("Fitting and transforming embeddings...")
    reduced_embeddings = reducer_instance.fit_transform(all_data_np)

    plt.figure(figsize=(12, 10))  # Slightly larger for better readability
    plt.title(
        f"{reducer_type.upper()} of Speech vs. Text Embeddings (N={num_samples_loaded})"
    )
    plt.scatter(
        reduced_embeddings[:num_samples_loaded, 0],
        reduced_embeddings[:num_samples_loaded, 1],
        c="blue",
        marker="o",
        label="Speech Embeddings",
        alpha=0.7,
    )
    plt.scatter(
        reduced_embeddings[num_samples_loaded:, 0],
        reduced_embeddings[num_samples_loaded:, 1],
        c="red",
        marker="x",
        label="Text Embeddings",
        alpha=0.7,
    )

    if draw_lines:
        for i in range(num_samples_loaded):
            speech_point = reduced_embeddings[i]
            text_point = reduced_embeddings[num_samples_loaded + i]
            d = speech_point - text_point
            plt.arrow(
                text_point[0],
                text_point[1],
                d[0],
                d[1],
                alpha=0.3,
                color="gray",
                head_width=0.01,
                length_includes_head=True,
            )

    plt.xlabel(f"{reducer_type.upper()} Dimension 1")
    plt.ylabel(f"{reducer_type.upper()} Dimension 2")
    plt.legend()
    plt.grid(True)
    if show_plots:
        plt.show()
    else:
        plot_filename = (
            output_dir / f"{reducer_type}_comparison_N{num_samples_loaded}.png"
        )
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
    plt.close()

    if use_ot_sampler and num_samples_loaded > 1:
        print("\n--- Applying OTPlanSampler (Optional) ---")
        # For 'exact', ensure num_samples_loaded is not too large, or it can be very slow.
        ot_method = "exact" if num_samples_loaded <= 2000 else "sinkhorn"  # Heuristic
        print(f"Using OT method: {ot_method} (N={num_samples_loaded})")
        ot_sampler = OTPlanSampler(method=ot_method)

        speech_tensor = torch.tensor(speech_data_np, dtype=torch.float32)
        text_tensor = torch.tensor(text_data_np, dtype=torch.float32)
        # Dummy labels for OT, as we want to see if it recovers the implicit pairing
        labels_speech = torch.arange(num_samples_loaded)
        labels_text = torch.arange(num_samples_loaded)

        try:
            # Note: OTPlanSampler might reorder the output based on its internal matching.
            # The returned labels label_text_ot and label_speech_ot indicate the *original* indices
            # of the items in text_tensor and speech_tensor that were matched.
            # text_data_ot will be text_tensor[label_text_ot]
            # speech_data_ot will be speech_tensor[label_speech_ot]
            text_data_ot, speech_data_ot, label_text_ot, label_speech_ot = (
                ot_sampler.sample_plan_with_labels(
                    text_tensor,
                    speech_tensor,
                    labels_text,
                    labels_speech,
                    replace=False,
                )
            )
            print(
                f"OT Sampler: Source indices matched (original text_tensor indices): {label_text_ot[:10]}..."
            )
            print(
                f"OT Sampler: Target indices matched (original speech_tensor indices): {label_speech_ot[:10]}..."
            )

            # Check how many times the OT sampler picked the *originally* corresponding pair
            # This means label_text_ot[i] should be equal to label_speech_ot[i] if the i-th pair
            # selected by OT was indeed an original (same-ID) pair.
            correct_matches_tensor = label_text_ot == label_speech_ot
            num_correct_matches = torch.sum(correct_matches_tensor).item()
            total_samples_ot = len(label_text_ot)  # Should be num_samples_loaded

            if total_samples_ot > 0:
                percent_correct_matches = (num_correct_matches / total_samples_ot) * 100
                print(
                    f"OT Sampler: Number of correct original pairings recovered: {num_correct_matches} / {total_samples_ot}"
                )
                print(
                    f"OT Sampler: Percentage of correct original pairings recovered: {percent_correct_matches:.2f}%"
                )
            else:
                print("OT Sampler: No samples processed by OT Sampler.")

            # For plotting, text_data_ot and speech_data_ot are already the matched pairs.
            # So text_data_ot[i] is matched with speech_data_ot[i].
            all_ot_data_np = torch.cat(
                (speech_data_ot, text_data_ot), dim=0
            ).numpy()  # speech first for consistency

            print("Fitting and transforming OT-sampled embeddings...")
            reduced_ot_embeddings = reducer_instance.fit_transform(
                all_ot_data_np
            )  # Use the same reducer instance

            plt.figure(figsize=(12, 10))
            plt.title(
                f"{reducer_type.upper()} of OT-Sampled Speech vs. Text (N={num_samples_loaded})"
            )
            # speech_data_ot is at the beginning of all_ot_data_np
            plt.scatter(
                reduced_ot_embeddings[:num_samples_loaded, 0],
                reduced_ot_embeddings[:num_samples_loaded, 1],
                c="blue",
                marker="s",
                label="Speech (OT-Sampled)",
                alpha=0.7,
            )  # Changed marker
            # text_data_ot is after speech_data_ot
            plt.scatter(
                reduced_ot_embeddings[num_samples_loaded:, 0],
                reduced_ot_embeddings[num_samples_loaded:, 1],
                c="red",
                marker="P",
                label="Text (OT-Sampled)",
                alpha=0.7,
            )  # Changed marker

            if draw_lines:
                for i in range(num_samples_loaded):
                    # reduced_ot_embeddings[i] corresponds to an element from speech_data_ot
                    # reduced_ot_embeddings[num_samples_loaded + i] corresponds to an element from text_data_ot
                    # These are already paired by the OT sampler.
                    speech_point_ot = reduced_ot_embeddings[i]
                    text_point_ot = reduced_ot_embeddings[num_samples_loaded + i]
                    d = speech_point_ot - text_point_ot
                    plt.arrow(
                        text_point_ot[0],
                        text_point_ot[1],
                        d[0],
                        d[1],
                        alpha=0.3,
                        color="green",
                        head_width=0.01,
                        length_includes_head=True,
                    )

            plt.xlabel(f"{reducer_type.upper()} Dimension 1")
            plt.ylabel(f"{reducer_type.upper()} Dimension 2")
            plt.legend()
            plt.grid(True)
            if show_plots:
                plt.show()
            else:
                ot_plot_filename = (
                    output_dir
                    / f"{reducer_type}_OT_comparison_N{num_samples_loaded}.png"
                )
                plt.savefig(ot_plot_filename)
                print(f"OT Sampler plot saved to {ot_plot_filename}")
            plt.close()

        except Exception as e:
            print(f"Error during OTPlanSampler processing: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare speech vs. text embeddings from two directories."
    )

    parser.add_argument(
        "speech_dir",
        type=str,
        help="Path to the directory containing speech embedding (.pt) files.",
    )
    parser.add_argument(
        "text_dir",
        type=str,
        help="Path to the directory containing text embedding (.pt) files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional. Directory to save output plots. Defaults to saving in the speech_dir.",
    )

    analysis_choices = ["l2_norm", "visualize"]
    parser.add_argument(
        "--analysis",
        type=str.lower,
        default="l2_norm",
        choices=analysis_choices,
        help=f"Type of analysis to compute (choices: {', '.join(analysis_choices)}; default: l2_norm).",
    )

    # Arguments specific to 'visualize' analysis
    parser.add_argument(
        "--reducer",
        type=str.lower,
        default="umap",
        choices=["umap", "tsne", "pca"],
        help="Dimensionality reduction technique for visualization (default: umap). Only used if analysis is 'visualize'.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=100,
        help="Maximum number of common samples to use for visualization (default: 100). Only used if analysis is 'visualize'.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable displaying generated plots, save them to files instead. Only used if analysis is 'visualize'.",
    )
    parser.add_argument(
        "--no-lines",
        action="store_true",
        help="Do not draw lines connecting speech/text pairs in the scatter plot. Only used if analysis is 'visualize'.",
    )
    parser.add_argument(
        "--use-ot-sampler",
        action="store_true",
        help="Additionally use OTPlanSampler and generate a plot for its output in 'visualize' analysis (experimental).",
    )

    args = parser.parse_args()

    speech_path = Path(args.speech_dir)
    text_path = Path(args.text_dir)
    output_path = (
        Path(args.output_dir) if args.output_dir else speech_path
    )  # Default to speech_path for output if not specified

    if args.analysis == "l2_norm":
        l2_norm(
            speech_dir=speech_path,
            text_dir=text_path,
        )
    elif args.analysis == "visualize":
        visualize_embeddings(
            speech_dir=speech_path,
            text_dir=text_path,
            reducer_type=args.reducer,
            N=args.N,
            show_plots=not args.no_plots,
            draw_lines=not args.no_lines,
            use_ot_sampler=args.use_ot_sampler,
            output_dir=output_path,
        )
    else:
        print(
            f"Unknown analysis type: {args.analysis}. Supported: {', '.join(analysis_choices)}"
        )


if __name__ == "__main__":
    main()
