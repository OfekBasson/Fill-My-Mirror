"""
Constrained-Pixels / Mirror-Pixels Ratio on MirrorBench V2
============================================================
For each index in mirrorbench_v2/gt_geometry/, downloads:
  Constrained mask : <idx>/constrained_pixels_gt_geometry_mask.png
  Mirror mask      : <idx>/generative_refinement_mask.png

and computes ratio = constrained_pixels / mirror_pixels, i.e. what
fraction of the mirror region is "constrained". Reports the average
ratio across all indices.

Usage
-----
    conda run -n fill-my-mirror python scripts/compute_constrained_pixels_ratio.py \
        --indices 0 1 2

    # Run all indices:
    conda run -n fill-my-mirror python scripts/compute_constrained_pixels_ratio.py \
        --output-csv outputs/constrained_pixels_ratio.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fill_my_mirror.storage import R2Client

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

R2_PREFIX = "mirrorbench_v2/gt_geometry"


def _download_pil(r2: R2Client, key: str) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        r2.download_file(key, Path(tf.name))
        return Image.open(tf.name).copy()


def _pil_to_binary(pil_img: Image.Image) -> np.ndarray:
    return np.asarray(pil_img.convert("L"), dtype=np.uint8) > 127


def compute_ratio_for_sample(idx: int, r2: R2Client) -> dict | None:
    prefix = f"{R2_PREFIX}/{idx}"
    constrained_key = f"{prefix}/constrained_pixels_gt_geometry_mask.png"
    mirror_key = f"{prefix}/generative_refinement_mask.png"

    try:
        constrained_pil = _download_pil(r2, constrained_key)
        mirror_pil = _download_pil(r2, mirror_key)
    except Exception as e:
        logger.warning("[%d] Failed to download R2 assets: %s", idx, e)
        return None

    constrained_mask = _pil_to_binary(constrained_pil)
    mirror_mask = _pil_to_binary(mirror_pil)

    n_constrained = int(constrained_mask.sum())
    n_mirror = int(mirror_mask.sum())
    ratio = n_constrained / n_mirror if n_mirror > 0 else float("nan")

    logger.info("[%d] constrained=%d  mirror=%d  ratio=%.4f", idx, n_constrained, n_mirror, ratio)

    return {
        "idx": idx,
        "n_constrained_px": n_constrained,
        "n_mirror_px": n_mirror,
        "ratio": ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute constrained_pixels / mirror_pixels ratio on MirrorBench V2."
    )
    parser.add_argument(
        "--indices", type=int, nargs="*", default=None,
        help="Subset of dataset indices. Default: all available.",
    )
    parser.add_argument(
        "--output-csv", type=str, default="outputs/constrained_pixels_ratio.csv",
        help="Path for per-sample results CSV.",
    )
    args = parser.parse_args()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    r2 = R2Client()

    if args.indices is not None:
        indices = args.indices
    else:
        logger.info("Listing available indices from R2 under %s ...", R2_PREFIX)
        keys = r2.list_keys(R2_PREFIX + "/")
        seen: set[int] = set()
        for k in keys:
            parts = k.removeprefix(R2_PREFIX + "/").split("/")
            if parts[0].isdigit():
                seen.add(int(parts[0]))
        indices = sorted(seen)
        logger.info("Found %d indices.", len(indices))

    rows: list[dict] = []
    for idx in indices:
        result = compute_ratio_for_sample(idx, r2)
        if result is not None:
            rows.append(result)

    if not rows:
        logger.error("No results produced.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    logger.info("Saved per-sample results to %s", output_csv)

    mean_ratio = df["ratio"].mean()
    print("\n" + "=" * 60)
    print("Constrained-Pixels / Mirror-Pixels Ratio on MirrorBench V2")
    print(f"  Samples evaluated : {len(df)}")
    print(f"  Average ratio     : {mean_ratio:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
