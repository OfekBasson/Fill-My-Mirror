"""
Batch inpainting script for Fill My Mirror.

Runs the full pipeline on every sample in the chosen HuggingFace dataset and
saves results as ``{output_dir}/seed_{seed}/{index}.png``.

Examples
--------
Run on the real-images dataset::

    python scripts/run_batch.py --dataset real --output-dir outputs/batch_real/

Run on the Blender dataset::

    python scripts/run_batch.py --dataset blender --output-dir outputs/batch_blender/

Process only a subset (indices 0–9)::

    python scripts/run_batch.py --dataset real --output-dir outputs/batch_real/ \\
        --start-index 0 --end-index 10

Resume an interrupted run (skip already-generated images)::

    python scripts/run_batch.py --dataset real --output-dir outputs/batch_real/ \\
        --skip-existing

The output directory layout is compatible with ``scripts/evaluate.py batch``::

    python scripts/evaluate.py batch \\
        --results-dir outputs/batch_real/seed_0/ \\
        --dataset real \\
        --output-dir outputs/eval/real/
"""

import argparse
import logging
from pathlib import Path

import yaml

from fill_my_mirror.geometry import estimate_geometry
from fill_my_mirror.projection import run_projection
from fill_my_mirror.dual_mask_inpainting import load_inpainting_pipeline, run_dual_mask_inpainting
from fill_my_mirror.loaders import RealImageSampleLoader, BlenderSampleLoader
from fill_my_mirror.utils import check_and_fix_aspect_ratio

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run Fill My Mirror inpainting on all samples in a dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- required ---
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["real", "blender"],
        help="Which HuggingFace dataset to use ('real' or 'blender').",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Root directory for outputs. Results are saved to {output_dir}/seed_{seed}/{index}.png.",
    )

    # --- optional control ---
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="First dataset index to process (inclusive, default: 0).",
    )
    parser.add_argument(
        "--end-index", type=int, default=None,
        help="Last dataset index to process (exclusive, default: full dataset).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=False,
        help="Skip indices whose output file already exists (useful for resuming).",
    )

    # --- pipeline pass-through ---
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-2", type=str, default=None)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=30.0)
    parser.add_argument("--num-images-per-prompt", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=float, default=1024)
    parser.add_argument("--width", type=float, default=1024)
    parser.add_argument("--n", type=float, default=6.0)
    parser.add_argument("--t-prime", type=float, default=750.0)
    parser.add_argument(
        "--blender_path", type=str, default=None,
        help="Path to the Blender executable. Overrides the config file value.",
    )

    args = parser.parse_args()

    config = load_config(Path(args.config))

    blender_path = Path(args.blender_path) if args.blender_path is not None else Path(config["blender_path"])
    if not blender_path.exists():
        raise FileNotFoundError(
            f"Blender was not found at: {blender_path}\n"
            "Please install it first with: bash scripts/install_blender.sh"
        )

    if args.dataset == "real":
        loader = RealImageSampleLoader(config["hf_dataset_repo"])
    else:
        loader = BlenderSampleLoader(config["hf_blender_dataset_repo"])

    dataset_size = len(loader)
    start = args.start_index
    end = args.end_index if args.end_index is not None else dataset_size

    if start < 0 or start >= dataset_size:
        parser.error(f"--start-index {start} is out of range (dataset has {dataset_size} samples).")
    if end > dataset_size:
        logger.warning(
            "--end-index %d exceeds dataset size %d; clamping to %d.",
            end, dataset_size, dataset_size,
        )
        end = dataset_size

    seed_dir = Path(args.output_dir) / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    total = end - start
    print(f"Dataset : {args.dataset} ({dataset_size} samples)")
    print(f"Range   : [{start}, {end})")
    print(f"Output  : {seed_dir}")
    print()

    print("Loading inpainting pipeline ...")
    pipe = load_inpainting_pipeline(model_name=config["inpainting_model_name"])
    print("Pipeline loaded.\n")

    for i, index in enumerate(range(start, end)):
        out_path = seed_dir / f"{index}.png"

        if args.skip_existing and out_path.exists():
            print(f"[{i + 1}/{total}] index {index} — skipping (already exists)")
            continue

        print(f"[{i + 1}/{total}] index {index} — processing ...")

        sample = loader.load(index)
        prompt = args.prompt or sample.prompt or config["prompt"]
        width = check_and_fix_aspect_ratio(sample.image_path, int(args.height), int(args.width))

        geometry = estimate_geometry(sample, config["geometry_model_name"])

        projection = run_projection(
            geometry_output=geometry,
            image_path=sample.image_path,
            mirror_mask_path=sample.mask_path,
            blender_path=blender_path,
        )

        run_dual_mask_inpainting(
            prompt=prompt,
            projected_image_path=projection.projected_image_path,
            geometry_constraint_mask_path=projection.geometry_constraint_mask_path,
            generative_refinement_mask_path=sample.mask_path,
            output_path=out_path,
            model_name=config["inpainting_model_name"],
            prompt_2=args.prompt_2,
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt,
            max_sequence_length=args.max_sequence_length,
            seed=args.seed,
            height=args.height,
            width=width,
            n=args.n,
            t_prime=args.t_prime,
            pipe=pipe,
        )

        print(f"[{i + 1}/{total}] index {index} — saved to {out_path}")

    print(f"\nDone. Results in {seed_dir}")


if __name__ == "__main__":
    main()
