"""
Projection experiment script for Fill My Mirror.

Runs geometry estimation + projection for every sample in a dataset and saves
structured outputs locally, then uploads to Cloudflare R2.

Output layout per image:
    <output_root>/<dataset>/<geometry_subdir>/<index>/
        original_image.png
        gt_image.png
        generative_refinement_mask.png
        projected_image.png
        geometry_constraint_mask.png
        prompt.json
        timing.json
        error.txt          # only on failure (full traceback)

R2 summary (updated after every upload batch):
    <dataset>/<geometry_subdir>/errors.txt

Examples
--------
Run on the MirrorBench dataset:

    python scripts/run_projection_experiment.py \\
        --dataset mirrorbench_v2 \\
        --output-root /tmp/proj_experiment

Process a subset and skip already-uploaded samples:

    python scripts/run_projection_experiment.py \\
        --dataset blender --output-root /tmp/proj_experiment \\
        --start-index 0 --end-index 50 --skip-existing
"""

import argparse
import gc
import json
import logging
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from fill_my_mirror.geometry import estimate_geometry
from fill_my_mirror.loaders import (
    BlenderSampleLoader,
    MirrorBenchV2SampleLoader,
    RealImageSampleLoader,
)
from fill_my_mirror.projection import run_projection_single_mirror
from fill_my_mirror.storage import R2Client

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
DEFAULT_UPLOAD_EVERY = 15

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _geometry_subdir(use_estimated: bool) -> str:
    return "estimated_geometry" if use_estimated else "gt_geometry"


def _r2_done_key(dataset: str, geom_subdir: str, index: int) -> str:
    return f"{dataset}/{geom_subdir}/{index}/timing.json"


def _upload_errors(
    r2: R2Client,
    errors: dict,
    output_root: Path,
    dataset: str,
    geom_subdir: str,
) -> None:
    if not errors:
        return
    summary_path = output_root / dataset / geom_subdir / "errors.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{idx}: {desc}" for idx, desc in sorted(errors.items())]
    summary_path.write_text("\n".join(lines) + "\n")
    r2.upload_file(summary_path, f"{dataset}/{geom_subdir}/errors.txt")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run geometry+projection on a dataset and upload outputs to R2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", required=True, choices=["real", "blender", "mirrorbench_v2"],
        help="Dataset to process.",
    )
    parser.add_argument(
        "--output-root", required=True, type=str,
        help="Local root directory for outputs.",
    )
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="First index to process (inclusive).",
    )
    parser.add_argument(
        "--end-index", type=int, default=None,
        help="Last index to process (exclusive, default: full dataset).",
    )
    parser.add_argument(
        "--upload-every", type=int, default=DEFAULT_UPLOAD_EVERY,
        help=f"Upload completed dirs to R2 every N images (default: {DEFAULT_UPLOAD_EVERY}).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=False,
        help="Skip samples whose timing.json already exists in R2.",
    )
    parser.add_argument(
        "--use-estimated-geometry", action="store_true", default=False,
        help=(
            "Estimate geometry from the image instead of using ground-truth geometry. "
            "Only relevant for --dataset blender or mirrorbench_v2."
        ),
    )
    parser.add_argument(
        "--blender-path", type=str, default=None,
        help="Path to the Blender executable. Overrides the config file value.",
    )

    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_root = Path(args.output_root)

    blender_path = Path(args.blender_path) if args.blender_path else Path(config["blender_path"])
    if not blender_path.exists():
        raise FileNotFoundError(
            f"Blender not found at: {blender_path}\n"
            "Install it with: bash scripts/install_blender.sh"
        )

    if args.dataset == "real":
        loader = RealImageSampleLoader()
    elif args.dataset == "blender":
        loader = BlenderSampleLoader()
    else:
        loader = MirrorBenchV2SampleLoader()

    dataset_size = len(loader)
    start = args.start_index
    end = args.end_index if args.end_index is not None else dataset_size

    if start < 0 or start >= dataset_size:
        parser.error(f"--start-index {start} out of range (dataset has {dataset_size} samples).")
    if end > dataset_size:
        logger.warning("--end-index %d exceeds dataset size %d; clamping.", end, dataset_size)
        end = dataset_size

    r2 = R2Client()
    total = end - start
    pending_upload: list[Path] = []
    errors: dict[int, str] = {}  # index -> short error description (one line)
    use_estimated = True if args.dataset == "real" else args.use_estimated_geometry
    geom_subdir = _geometry_subdir(use_estimated)

    print(f"Dataset  : {args.dataset} ({dataset_size} samples)")
    print(f"Geometry : {geom_subdir}")
    print(f"Range    : [{start}, {end})")
    print(f"Output   : {output_root}")
    print()

    for i, index in enumerate(range(start, end)):
        label = f"[{i + 1}/{total}] index {index}"

        if args.skip_existing and r2.key_exists(_r2_done_key(args.dataset, geom_subdir, index)):
            print(f"{label} — skipping (already in R2)")
            continue

        print(f"{label} — loading ...")
        sample = loader.load(index, use_estimated_geometry=use_estimated)
        prompt = sample.prompt or config["prompt"]
        out_dir = output_root / args.dataset / geom_subdir / str(index)
        out_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(sample.image_path, out_dir / "original_image.png")
        if sample.gt_image_path:
            shutil.copy(sample.gt_image_path, out_dir / "gt_image.png")
        shutil.copy(sample.mask_path, out_dir / "generative_refinement_mask.png")
        (out_dir / "prompt.json").write_text(json.dumps({"prompt": prompt}))

        t_start = time.perf_counter()
        try:
            print(f"{label} — estimating geometry ...")
            geometry = estimate_geometry(sample, config["geometry_model_name"], tmp_dir=out_dir)
            if hasattr(geometry, "depth") and geometry.depth is not None:
                np.save(out_dir / "depth.npy", geometry.depth)
            gc.collect()
            torch.cuda.empty_cache()

            print(f"{label} — running projection ...")
            projection = run_projection_single_mirror(
                geometry_output=geometry,
                image_path=sample.image_path,
                mirror_mask_path=sample.mask_path,
                blender_path=blender_path,
                projected_image_path=out_dir / "projected_image.png",
                geometry_constraint_mask_path=out_dir / "geometry_constraint_mask.png",
                tmp_dir=out_dir,
            )

            t_elapsed = time.perf_counter() - t_start
            (out_dir / "timing.json").write_text(json.dumps({"projection_time_seconds": round(t_elapsed, 2)}))

            print(f"{label} — done ({t_elapsed:.1f}s), saved to {out_dir}, queued for upload")

        except Exception as exc:
            t_elapsed = time.perf_counter() - t_start
            short_desc = f"{type(exc).__name__}: {exc}"
            print(f"{label} — ERROR after {t_elapsed:.1f}s: {short_desc}")

            (out_dir / "error.txt").write_text(traceback.format_exc())
            shutil.copy(sample.image_path, out_dir / "projected_image.png")
            shutil.copy(sample.mask_path, out_dir / "geometry_constraint_mask.png")
            (out_dir / "timing.json").write_text(
                json.dumps({"projection_time_seconds": round(t_elapsed, 2), "error": True})
            )
            errors[index] = short_desc

        pending_upload.append(out_dir)

        if len(pending_upload) >= args.upload_every:
            _flush_uploads(r2, pending_upload, args.dataset, geom_subdir)
            pending_upload.clear()
            _upload_errors(r2, errors, output_root, args.dataset, geom_subdir)

    if pending_upload:
        _flush_uploads(r2, pending_upload, args.dataset, geom_subdir)
        _upload_errors(r2, errors, output_root, args.dataset, geom_subdir)

    if errors:
        print(f"\n{len(errors)} error(s):")
        for idx, desc in sorted(errors.items()):
            print(f"  [{idx}] {desc}")

    print(f"\nDone. Outputs in {output_root}")


def _flush_uploads(r2: R2Client, dirs: list[Path], dataset: str, geom_subdir: str) -> None:
    for out_dir in dirs:
        index = out_dir.name
        r2_prefix = f"{dataset}/{geom_subdir}/{index}"
        print(f"  Uploading {index} → R2:{r2_prefix}/")
        r2.upload_dir(out_dir, r2_prefix)


if __name__ == "__main__":
    main()
