"""
GT-vs-shifted-GT sanity check for the real-images evaluation pipeline.

Evaluates each ground-truth image against a copy of itself shifted by N pixels
to the right (edge-replicated, default N=1), using the RCS masks already
cached in R2 (does not recompute them — indices without a cached mask are
skipped). This measures how sensitive the metrics pipeline is to a small
pixel-level misalignment — a more realistic lower bound than exact GT-vs-GT.

Always uses the horizontal-flip-only RCS mask (rcs_mask_horizontal_only.png).

R2 paths read per index (same layout as scripts/run_real_evaluation.py):
  real/estimated_geometry/<idx>/gt_image.png
  real/estimated_geometry/<idx>/generative_refinement_mask.png
  real/estimated_geometry/<idx>/rcs_mask_horizontal_only.png

Outputs are written locally to --output-dir AND uploaded to R2 under
real/evaluation_gt_sanity_check/:
  <index>_metrics.json  (per index)
  per_index_metrics.csv
  aggregate.csv

Usage
-----
    python scripts/evaluate_gt_sanity_check.py
    python scripts/evaluate_gt_sanity_check.py --shift-pixels 2
    python scripts/evaluate_gt_sanity_check.py --output-dir outputs/gt_sanity_check
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fill_my_mirror.evaluation.metrics_computation import GeneratedImage, MetricsInput, compute_metrics
from fill_my_mirror.storage import R2Client

R2_BASE_PREFIX = "real/estimated_geometry"
R2_EVAL_PREFIX = "real/evaluation_gt_sanity_check"
RCS_FILENAME = "rcs_mask_horizontal_only.png"


def shift_right(image: Image.Image, pixels: int) -> Image.Image:
    """Shift an image `pixels` columns to the right, replicating the left edge column."""
    arr = np.array(image)
    shifted = np.empty_like(arr)
    shifted[:, pixels:] = arr[:, :-pixels]
    shifted[:, :pixels] = arr[:, :1]
    return Image.fromarray(shifted)


def discover_indices_with_gt(r2: R2Client) -> list[int]:
    keys = r2.list_keys(f"{R2_BASE_PREFIX}/")
    indices = []
    for key in keys:
        parts = key.split("/")
        if len(parts) == 4 and parts[-1] == "gt_image.png" and parts[-2].isdigit():
            indices.append(int(parts[-2]))
    return sorted(indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GT-vs-GT sanity check using cached hflip-only RCS masks")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gt_sanity_check"))
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--shift-pixels", type=int, default=1,
                        help="Number of pixels to shift the comparison image right (default: 1).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    r2 = R2Client()
    indices = discover_indices_with_gt(r2)
    if args.start_index is not None or args.end_index is not None:
        indices = [
            i for i in indices
            if (args.start_index is None or i >= args.start_index)
            and (args.end_index is None or i < args.end_index)
        ]
    print(f"Found {len(indices)} indices with a GT image.")

    rows = []
    with tempfile.TemporaryDirectory(prefix="gt_sanity_") as tmp_str:
        tmp_root = Path(tmp_str)

        for i, index in enumerate(indices):
            d = tmp_root / str(index)
            d.mkdir(parents=True, exist_ok=True)

            rcs_key = f"{R2_BASE_PREFIX}/{index}/{RCS_FILENAME}"
            if not r2.key_exists(rcs_key):
                print(f"  [{index}] no cached {RCS_FILENAME} in R2 — skipping (not recomputing).")
                continue

            gt_local = d / "gt_image.png"
            mirror_local = d / "generative_refinement_mask.png"
            rcs_local = d / RCS_FILENAME
            try:
                r2.download_file(f"{R2_BASE_PREFIX}/{index}/gt_image.png", gt_local)
                r2.download_file(f"{R2_BASE_PREFIX}/{index}/generative_refinement_mask.png", mirror_local)
                r2.download_file(rcs_key, rcs_local)
            except Exception as e:
                print(f"  [{index}] failed to download inputs: {e} — skipping.")
                continue

            gt_image = Image.open(gt_local).convert("RGB")
            shifted_image = shift_right(gt_image, args.shift_pixels)
            full_mirror_mask = Image.open(mirror_local).convert("L")
            constrained_mask = Image.open(rcs_local).convert("L")

            df = compute_metrics(MetricsInput(
                gt_image=gt_image,
                generated_images=[GeneratedImage(name=f"gt_shifted_{args.shift_pixels}px", image=shifted_image)],
                full_mirror_mask=full_mirror_mask,
                constrained_mask=constrained_mask,
                save_path=d,
                prompt="",
            ))
            metrics = df.iloc[0].drop("name").to_dict()
            metrics["index"] = index
            rows.append(metrics)
            metrics_local = args.output_dir / f"{index}_metrics.json"
            metrics_local.write_text(json.dumps(metrics, indent=2))
            r2.upload_file(metrics_local, f"{R2_EVAL_PREFIX}/{index}_metrics.json")

            if (i + 1) % 25 == 0 or (i + 1) == len(indices):
                print(f"  [{i + 1}/{len(indices)}] evaluated")

    if not rows:
        print("No indices evaluated (no cached RCS masks found).")
        return

    combined = pd.DataFrame(rows)
    combined_path = args.output_dir / "per_index_metrics.csv"
    combined.to_csv(combined_path, index=False)
    r2.upload_file(combined_path, f"{R2_EVAL_PREFIX}/per_index_metrics.csv")

    aggregate = combined.drop(columns=["index"]).agg(["mean", "std"]).T
    aggregate.index.name = "metric"
    aggregate_path = args.output_dir / "aggregate.csv"
    aggregate.to_csv(aggregate_path)
    r2.upload_file(aggregate_path, f"{R2_EVAL_PREFIX}/aggregate.csv")

    print(f"\nEvaluated {len(rows)}/{len(indices)} indices.")
    print(f"\n--- Aggregate (GT vs GT shifted right by {args.shift_pixels}px) ---")
    print(aggregate.to_string())
    print(f"\nSaved locally to {combined_path}, {aggregate_path}")
    print(f"Uploaded to R2 under {R2_EVAL_PREFIX}/")


if __name__ == "__main__":
    main()
