"""
Re-dilate existing RCS correspondence masks at 800×800 resolution.

For each sample in outputs/rcs_mirrorbench_eval/sample_<idx>/ that has a
saved combined_correspondence_mask.png, this script:
  1. Loads combined_correspondence_mask.png (original resolution).
  2. Loads generative_refinement_mask.png (mirror mask, original resolution).
     Falls back to R2 download if the local file doesn't exist.
  3. Resizes both to 800×800.
  4. Applies dilation (radius=4, iterations=1) at 800×800.
  5. Resizes the resulting RCS mask back to original resolution.
  6. Computes precision / recall / F1 against constrained_pixels_gt_geometry_mask.png.

Outputs:
  metrics_redilated.csv   — per-sample P/R/F1
  ranking_redilated.csv   — sorted by F1 descending
  precision_recall_scatter_redilated.pdf

Usage
-----
    python scripts/redilate_rcs_mirrorbench.py \
        --eval-dir outputs/rcs_mirrorbench_eval \
        --output-dir outputs/rcs_mirrorbench_eval

    # Specific indices only:
    python scripts/redilate_rcs_mirrorbench.py --indices 0 5 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fill_my_mirror.loaders import MirrorBenchV2SampleLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RCS_COMPUTE_SIZE = 800
DILATION_RADIUS = 4
DILATION_ITERATIONS = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dilate_and_intersect(corr: np.ndarray, mirror: np.ndarray) -> np.ndarray:
    kernel_size = 2 * DILATION_RADIUS + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(corr.astype(np.uint8) * 255, kernel, iterations=DILATION_ITERATIONS) > 127
    return dilated & mirror


def _compute_pr(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _load_mask(path: Path) -> np.ndarray:
    """Load a PNG mask as a boolean array."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def _resize_mask(mask: np.ndarray, size: tuple[int, int], interp=Image.NEAREST) -> np.ndarray:
    """Resize boolean mask to (W, H) = size using PIL."""
    pil = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(pil.resize(size, interp)) > 127


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def process_sample(
    idx: int,
    sample_dir: Path,
    loader: MirrorBenchV2SampleLoader,
) -> dict | None:
    corr_path = sample_dir / "combined_correspondence_mask.png"
    gt_path   = sample_dir / "constrained_pixels_gt_geometry_mask.png"

    if not corr_path.exists():
        logger.warning("[%d] combined_correspondence_mask.png not found — skipping.", idx)
        return None
    if not gt_path.exists():
        logger.warning("[%d] constrained_pixels_gt_geometry_mask.png not found — skipping.", idx)
        return None

    corr_orig = _load_mask(corr_path)
    gt_mask   = _load_mask(gt_path)
    H, W = corr_orig.shape

    # Load mirror mask from the HDF5 dataset
    try:
        sample = loader.load(idx)
        mirror_orig = _load_mask(Path(sample.mask_path))
    except Exception as e:
        logger.warning("[%d] Cannot load mirror mask from dataset: %s", idx, e)
        return None

    S = RCS_COMPUTE_SIZE
    corr_800   = _resize_mask(corr_orig,   (S, S))
    mirror_800 = _resize_mask(mirror_orig, (S, S))

    rcs_800 = _dilate_and_intersect(corr_800, mirror_800)

    # Resize RCS mask back to original resolution
    rcs_orig = _resize_mask(rcs_800, (W, H))

    # Save updated rcs_mask next to the original
    Image.fromarray((rcs_orig.astype(np.uint8) * 255), mode="L").save(
        sample_dir / "rcs_mask_redilated.png"
    )

    precision, recall, f1 = _compute_pr(rcs_orig, gt_mask)
    logger.info("[%d] P=%.3f  R=%.3f  F1=%.3f", idx, precision, recall, f1)
    return {"idx": idx, "precision": precision, "recall": recall, "f1": f1,
            "n_rcs_px": int(rcs_orig.sum()), "n_gt_px": int(gt_mask.sum())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    ranking = df.sort_values("f1", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    ranking.to_csv(output_dir / "ranking_redilated.csv", index=False)

    mean_p  = df["precision"].mean()
    mean_r  = df["recall"].mean()
    mean_f1 = df["f1"].mean()
    std_p   = df["precision"].std()
    std_r   = df["recall"].std()
    std_f1  = df["f1"].std()

    summary_lines = [
        "=" * 60,
        f"RCS Re-dilation at {RCS_COMPUTE_SIZE}×{RCS_COMPUTE_SIZE}  "
        f"(r={DILATION_RADIUS}, i={DILATION_ITERATIONS})",
        f"  Samples evaluated : {len(df)}",
        f"  Mean Precision    : {mean_p:.4f}  (std={std_p:.4f})",
        f"  Mean Recall       : {mean_r:.4f}  (std={std_r:.4f})",
        f"  Mean F1           : {mean_f1:.4f}  (std={std_f1:.4f})",
        "=" * 60,
        "",
        "Top 10 (best F1):",
        ranking.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "Bottom 10 (worst F1):",
        ranking.tail(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "=" * 60,
    ]
    summary = "\n".join(summary_lines)
    print("\n" + summary)

    txt_path = output_dir / "metrics_redilated.txt"
    txt_path.write_text(summary + "\n")
    logger.info("Saved summary to %s", txt_path)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["recall"], df["precision"], alpha=0.5, s=15, color="steelblue")
    ax.axvline(mean_r, color="red",    linestyle="--", linewidth=1, label=f"mean R={mean_r:.3f}")
    ax.axhline(mean_p, color="orange", linestyle="--", linewidth=1, label=f"mean P={mean_p:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    scatter_path = output_dir / "precision_recall_scatter_redilated.pdf"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scatter plot to %s", scatter_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Re-dilate RCS masks at {RCS_COMPUTE_SIZE}×{RCS_COMPUTE_SIZE} "
                    f"(r={DILATION_RADIUS}, i={DILATION_ITERATIONS})."
    )
    parser.add_argument("--eval-dir", type=Path, default=Path("outputs/rcs_mirrorbench_eval"),
                        help="Directory containing sample_<idx>/ subdirs.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write CSV/plot outputs. Defaults to --eval-dir.")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="Subset of indices. Default: all sample_* dirs in --eval-dir.")
    parser.add_argument("--from-csv", action="store_true",
                        help="Skip re-dilation; load existing metrics_redilated.csv and regenerate outputs only.")
    args = parser.parse_args()

    eval_dir   = args.eval_dir
    output_dir = args.output_dir or eval_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        metrics_path = output_dir / "metrics_redilated.csv"
        if not metrics_path.exists():
            logger.error("--from-csv requires %s to exist.", metrics_path)
            sys.exit(1)
        df = pd.read_csv(metrics_path)
        logger.info("Loaded %d rows from %s", len(df), metrics_path)
        _save_outputs(df, output_dir)
        return

    if args.indices is not None:
        indices = args.indices
    else:
        indices = sorted(
            int(p.name.split("_")[1])
            for p in eval_dir.glob("sample_*")
            if p.is_dir() and p.name.split("_")[1].isdigit()
        )
        logger.info("Found %d sample dirs.", len(indices))

    print("Loading MirrorBench V2 dataset...")
    loader = MirrorBenchV2SampleLoader()
    rows: list[dict] = []

    for idx in indices:
        sample_dir = eval_dir / f"sample_{idx}"
        result = process_sample(idx, sample_dir, loader)
        if result is not None:
            rows.append(result)

    if not rows:
        logger.error("No results produced.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    metrics_path = output_dir / "metrics_redilated.csv"
    df.to_csv(metrics_path, index=False)
    logger.info("Saved metrics to %s", metrics_path)

    _save_outputs(df, output_dir)


if __name__ == "__main__":
    main()
