"""
Evaluation script for Fill My Mirror.

Two subcommands are available:

``local`` — evaluate locally provided files (no HuggingFace dataset):
    python scripts/evaluate.py local \\
        --generated outputs/result.png \\
        --gt data/real_images/gt_images/0.png \\
        --mask data/real_images/masks/0.png \\
        --save-dir eval/sample_0/ \\
        [--rcs-dilation 5] [--prompt "A bedroom with a standing mirror"]

``batch`` — evaluate a directory of results against the HuggingFace dataset:
    python scripts/evaluate.py batch \\
        --results-dir outputs/my_run/ \\
        --dataset real \\
        --output-dir outputs/eval/ \\
        [--config configs/config.yaml] [--rcs-dilation 5]

For ``batch`` the script automatically picks the constraint mask method:
  - ``--dataset real``           → Reflection Consistency Score via MASt3R
  - ``--dataset blender``        → GT geometry projection via Blender
  - ``--dataset mirrorbench_v2`` → GT geometry projection via Blender (same as blender)

Result PNGs in ``--results-dir`` are expected to be named ``{index}.png``.
"""

import argparse
import logging
from pathlib import Path

import yaml
import pandas as pd
from PIL import Image

from fill_my_mirror.evaluation import (
    compute_metrics,
    compute_rcs_mask,
    compute_gt_geometry_constraint_mask,
    MetricsInput,
    GeneratedImage,
)
from fill_my_mirror.loaders import RealImageSampleLoader, BlenderSampleLoader, MirrorBenchV2SampleLoader

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")

logger = logging.getLogger(__name__)


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# local subcommand
# ---------------------------------------------------------------------------

def _run_local(args: argparse.Namespace, config: dict) -> None:
    generated_path = Path(args.generated)
    gt_path = Path(args.gt)
    mask_path = Path(args.mask)
    save_dir = Path(args.save_dir)
    prompt = args.prompt or ""

    gt_image = Image.open(gt_path).convert("RGB")
    mirror_mask = Image.open(mask_path).convert("L")
    generated_image = Image.open(generated_path).convert("RGB")

    constrained_mask_path = compute_rcs_mask(
        gt_image=gt_image,
        mirror_mask=mirror_mask,
        dataset_type="provided_images",
        mask_stem=generated_path.stem,
        mast3r_model_name=config["mast3r_model_name"],
        dilation_radius=args.rcs_dilation,
    )
    constrained_mask = Image.open(constrained_mask_path).convert("L")

    metrics_input = MetricsInput(
        gt_image=gt_image,
        generated_images=[GeneratedImage(name=generated_path.stem, image=generated_image)],
        full_mirror_mask=mirror_mask,
        constrained_mask=constrained_mask,
        save_path=save_dir,
        prompt=prompt,
    )

    df = compute_metrics(metrics_input)
    print("\nMetrics:")
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# batch subcommand
# ---------------------------------------------------------------------------

def _run_batch(args: argparse.Namespace, config: dict) -> None:
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_type = args.dataset  # "real", "blender", or "mirrorbench_v2"
    use_rcs = dataset_type == "real"

    blender_path = Path(args.blender_path) if args.blender_path is not None else Path(config["blender_path"])

    if dataset_type == "real":
        loader = RealImageSampleLoader()
    elif dataset_type == "blender":
        loader = BlenderSampleLoader()
    else:
        loader = MirrorBenchV2SampleLoader()

    result_pngs = sorted(results_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not result_pngs:
        logger.error("No PNG files found in %s. Nothing to evaluate.", results_dir)
        return

    all_dfs: list[pd.DataFrame] = []

    for result_png in result_pngs:
        try:
            index = int(result_png.stem)
        except ValueError:
            logger.warning("Skipping '%s' — filename is not an integer index.", result_png.name)
            continue

        if index >= len(loader):
            logger.warning(
                "Skipping index %d — out of range (dataset has %d samples).",
                index, len(loader),
            )
            continue

        sample = loader.load(index)
        prompt = args.prompt or sample.prompt or config["prompt"]

        gt_image = Image.open(sample.gt_image_path).convert("RGB")
        mirror_mask = Image.open(sample.mask_path).convert("L")
        generated_image = Image.open(result_png).convert("RGB")

        sample_save_dir = output_dir / str(index)

        if use_rcs:
            constrained_mask_path = compute_rcs_mask(
                gt_image=gt_image,
                mirror_mask=mirror_mask,
                dataset_type="real_images",
                mask_stem=str(index),
                mast3r_model_name=config["mast3r_model_name"],
                dilation_radius=args.rcs_dilation,
            )
        else:
            # Blender: use GT geometry
            if not blender_path.exists():
                raise FileNotFoundError(
                    f"Blender executable not found at '{blender_path}'. "
                    "Set blender_path in the config or pass --blender-path."
                )
            constrained_mask_path = compute_gt_geometry_constraint_mask(
                sample=sample,
                blender_path=blender_path,
                mask_stem=str(index),
            )

        constrained_mask = Image.open(constrained_mask_path).convert("L")

        metrics_input = MetricsInput(
            gt_image=gt_image,
            generated_images=[GeneratedImage(name=result_png.stem, image=generated_image)],
            full_mirror_mask=mirror_mask,
            constrained_mask=constrained_mask,
            save_path=sample_save_dir,
            prompt=prompt,
        )

        df = compute_metrics(metrics_input)
        df["index"] = index
        all_dfs.append(df)
        print(f"[{index}/{len(result_pngs) - 1}] evaluated {result_png.name}")

    if not all_dfs:
        print("No samples were evaluated.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    aggregate = (
        combined.drop(columns=["name", "index"], errors="ignore")
        .agg(["mean", "std"])
        .T.rename(columns={"mean": "mean", "std": "std"})
    )
    aggregate.index.name = "metric"
    aggregate_path = output_dir / "aggregate.csv"
    aggregate.to_csv(aggregate_path)

    print("\n--- Aggregate results ---")
    print(aggregate.to_string())
    print(f"\nAggregate saved to {aggregate_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate Fill My Mirror outputs against ground-truth images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file.",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # -- local subcommand --
    local_parser = subparsers.add_parser(
        "local",
        help="Evaluate a single locally provided image against a GT image (no HuggingFace dataset).",
    )
    local_parser.add_argument(
        "--generated", type=str, required=True,
        help="Path to the generated/inpainted image.",
    )
    local_parser.add_argument(
        "--gt", type=str, required=True,
        help="Path to the ground-truth image.",
    )
    local_parser.add_argument(
        "--mask", type=str, required=True,
        help="Path to the binary mirror mask.",
    )
    local_parser.add_argument(
        "--save-dir", type=str, required=True,
        help="Directory to save the metrics CSV.",
    )
    local_parser.add_argument(
        "--rcs-dilation", type=int, default=5,
        help="Dilation radius for the RCS mask (default: 5).",
    )
    local_parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt for CLIP similarity.",
    )

    # -- batch subcommand --
    batch_parser = subparsers.add_parser(
        "batch",
        help="Evaluate a directory of result images against the HuggingFace dataset.",
    )
    batch_parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing result PNGs named {index}.png.",
    )
    batch_parser.add_argument(
        "--dataset", type=str, required=True, choices=["real", "blender", "mirrorbench_v2"],
        help="Which dataset to load ground truth from ('real', 'blender', or 'mirrorbench_v2').",
    )
    batch_parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save per-sample and aggregate CSVs.",
    )
    batch_parser.add_argument(
        "--rcs-dilation", type=int, default=5,
        help="Dilation radius for the RCS mask (default: 5).",
    )
    batch_parser.add_argument(
        "--prompt", type=str, default=None,
        help="Override CLIP prompt for all samples.",
    )
    batch_parser.add_argument(
        "--blender-path", type=str, default=None,
        help="Path to the Blender executable (needed for --dataset blender).",
    )

    args = parser.parse_args()
    config = _load_config(Path(args.config))

    if args.subcommand == "local":
        _run_local(args, config)
    else:
        _run_batch(args, config)


if __name__ == "__main__":
    main()
