"""
Compute generative_refinement_mask for every sample in the Blender or MirrorBench-V2
dataset by reading projection outputs already stored in R2.

    generative_refinement_mask = mirror_mask & ~geometry_constraint_mask

The geometry_constraint_mask and mirror mask (generative_refinement_mask.png) are
downloaded from R2 at:
    {dataset}/{geom_subdir}/{index}/geometry_constraint_mask.png
    {dataset}/{geom_subdir}/{index}/generative_refinement_mask.png

Usage:
    python scripts/compute_generative_refinement_masks.py \
        --dataset mirrorbench_v2 \
        --output-dir outputs/refinement_masks/mirrorbench_v2/
"""

from __future__ import annotations

import argparse
import tempfile
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from fill_my_mirror.storage import R2Client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute generative refinement masks from R2 projection outputs")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["blender", "mirrorbench_v2"],
    )
    parser.add_argument(
        "--geom-subdir",
        default="gt_geometry",
        choices=["gt_geometry", "estimated_geometry"],
        help="Geometry subdirectory used when running the projection experiment",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    r2 = R2Client()

    # Discover available indices from R2 if end-index not given
    prefix = f"{args.dataset}/{args.geom_subdir}/"
    if args.end_index is None:
        keys = r2.list_keys(prefix)
        indices = sorted({int(k.split("/")[2]) for k in keys if k.split("/")[2].isdigit()})
        indices = [i for i in indices if i >= args.start_index]
    else:
        indices = list(range(args.start_index, args.end_index))

    total = len(indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset  : {args.dataset}/{args.geom_subdir}")
    print(f"Indices  : {total}")
    print(f"Output   : {args.output_dir}")
    print()

    for i, index in enumerate(indices):
        label = f"[{i + 1}/{total}] index {index}"
        out_path = args.output_dir / f"{index}.png"

        if args.skip_existing and out_path.exists():
            print(f"{label} — skipping")
            continue

        try:
            base = f"{args.dataset}/{args.geom_subdir}/{index}"

            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                r2.download_file(f"{base}/geometry_constraint_mask.png", tmp / "constraint.png")
                r2.download_file(f"{base}/generative_refinement_mask.png", tmp / "mirror.png")
                inpainting_mask = np.asarray(Image.open(tmp / "constraint.png").convert("L")) > 127
                mirror_mask = np.asarray(Image.open(tmp / "mirror.png").convert("L")) > 127

            refinement_mask = mirror_mask & ~inpainting_mask
            Image.fromarray((refinement_mask.astype(np.uint8) * 255), mode="L").save(out_path)
            print(f"{label} — saved")

        except Exception:
            print(f"{label} — ERROR")
            traceback.print_exc()


if __name__ == "__main__":
    main()
