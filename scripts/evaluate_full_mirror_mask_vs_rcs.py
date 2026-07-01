"""
Full Mirror Mask vs RCS Mask Evaluation
========================================
Reads precomputed masks from a local RCS eval output directory and computes
precision/recall for BOTH the full mirror mask and the RCS mask against the
ground-truth constrained-pixels mask.

No MAST3R or R2 required — just load files from disk.

Expected per-sample layout under <eval-dir>/sample_<idx>/:
  generative_refinement_mask.png         (full mirror mask)
  rcs_mask.png                           (RCS mask)
  constrained_pixels_gt_geometry_mask.png (GT)

Usage
-----
    # MirrorBench V2:
    conda run -n fill-my-mirror python scripts/evaluate_full_mirror_mask_vs_rcs.py --eval-dir outputs/rcs_mirrorbench_eval --output-dir outputs/full_vs_rcs_mirrorbench

    # Blender:
    conda run -n fill-my-mirror python scripts/evaluate_full_mirror_mask_vs_rcs.py \\
        --eval-dir outputs/rcs_blender_eval \\
        --output-dir outputs/full_vs_rcs_blender
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_pr(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _load_binary(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(idx: int, eval_dir: Path) -> dict | None:
    sample_dir = eval_dir / f"sample_{idx}"
    full_mask_path = sample_dir / "generative_refinement_mask.png"
    rcs_mask_path  = sample_dir / "rcs_mask.png"
    gt_mask_path   = sample_dir / "constrained_pixels_gt_geometry_mask.png"

    for p in (full_mask_path, rcs_mask_path, gt_mask_path):
        if not p.exists():
            logger.warning("[%d] Missing file: %s — skipping.", idx, p)
            return None

    full_mask = _load_binary(full_mask_path)
    rcs_mask  = _load_binary(rcs_mask_path)
    gt_mask   = _load_binary(gt_mask_path)

    # Resize everything to GT resolution if they differ (shouldn't, but be safe)
    H, W = gt_mask.shape
    if full_mask.shape != (H, W):
        full_mask = np.asarray(
            Image.fromarray(full_mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
        ) > 127
    if rcs_mask.shape != (H, W):
        rcs_mask = np.asarray(
            Image.fromarray(rcs_mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
        ) > 127

    full_p, full_r, full_f1 = _compute_pr(full_mask, gt_mask)
    rcs_p,  rcs_r,  rcs_f1  = _compute_pr(rcs_mask,  gt_mask)

    logger.info(
        "[%d] Full: P=%.3f R=%.3f F1=%.3f  |  RCS: P=%.3f R=%.3f F1=%.3f",
        idx, full_p, full_r, full_f1, rcs_p, rcs_r, rcs_f1,
    )

    return {
        "idx":            idx,
        "full_precision": full_p,
        "full_recall":    full_r,
        "full_f1":        full_f1,
        "full_n_px":      int(full_mask.sum()),
        "rcs_precision":  rcs_p,
        "rcs_recall":     rcs_r,
        "rcs_f1":         rcs_f1,
        "rcs_n_px":       int(rcs_mask.sum()),
        "n_gt_px":        int(gt_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare full mirror mask vs RCS mask precision/recall against GT."
    )
    parser.add_argument(
        "--eval-dir", type=str, required=True,
        help="Directory containing precomputed sample_<idx>/ folders (rcs_mask.png etc.).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output CSVs and plots. Defaults to <eval-dir>.",
    )
    parser.add_argument(
        "--indices", type=int, nargs="*", default=None,
        help="Subset of sample indices. Default: all sample_* directories found.",
    )
    args = parser.parse_args()

    eval_dir   = Path(args.eval_dir)
    output_dir = Path(args.output_dir) if args.output_dir else eval_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.indices is not None:
        indices = sorted(args.indices)
    else:
        indices = sorted(
            int(p.name.split("_")[1])
            for p in eval_dir.glob("sample_*")
            if p.is_dir() and p.name.split("_")[1].isdigit()
        )
        logger.info("Found %d sample directories.", len(indices))

    rows: list[dict] = []
    for idx in indices:
        result = evaluate_sample(idx, eval_dir)
        if result is not None:
            rows.append(result)

    if not rows:
        logger.error("No results produced.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)

    # ---- metrics.csv in same structure as existing scripts (for full mask) ----
    full_metrics = df[["idx"]].copy()
    full_metrics["precision"] = df["full_precision"]
    full_metrics["recall"]    = df["full_recall"]
    full_metrics["f1"]        = df["full_f1"]
    full_metrics["n_mask_px"] = df["full_n_px"]
    full_metrics["n_gt_px"]   = df["n_gt_px"]
    full_metrics_path = output_dir / "full_mask_metrics.csv"
    full_metrics.to_csv(full_metrics_path, index=False)
    logger.info("Saved full-mask metrics to %s", full_metrics_path)

    # ---- combined CSV with both ----
    combined_path = output_dir / "full_vs_rcs_metrics.csv"
    df.to_csv(combined_path, index=False)
    logger.info("Saved combined metrics to %s", combined_path)

    # ---- Summary ----
    mean_full_p  = df["full_precision"].mean()
    mean_full_r  = df["full_recall"].mean()
    mean_full_f1 = df["full_f1"].mean()
    mean_rcs_p   = df["rcs_precision"].mean()
    mean_rcs_r   = df["rcs_recall"].mean()
    mean_rcs_f1  = df["rcs_f1"].mean()

    print("\n" + "=" * 65)
    print(f"Full Mirror Mask vs RCS Mask  ({len(df)} samples)")
    print(f"{'':30s}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    print(f"{'Full mirror mask':30s}{mean_full_p:10.4f}{mean_full_r:10.4f}{mean_full_f1:10.4f}")
    print(f"{'RCS mask':30s}{mean_rcs_p:10.4f}{mean_rcs_r:10.4f}{mean_rcs_f1:10.4f}")
    print(f"{'Precision gain (RCS - Full)':30s}{mean_rcs_p - mean_full_p:+10.4f}")
    print("=" * 65)

    # ---- Scatter: full mask P/R ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["full_recall"], df["full_precision"], alpha=0.5, s=15, color="tomato",
               label="Full mirror mask")
    ax.axvline(mean_full_r, color="red",    linestyle="--", linewidth=1,
               label=f"mean R={mean_full_r:.3f}")
    ax.axhline(mean_full_p, color="orange", linestyle="--", linewidth=1,
               label=f"mean P={mean_full_p:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("Full Mirror Mask Precision–Recall", fontsize=10)
    fig.tight_layout()
    scatter_path = output_dir / "precision_recall_scatter.pdf"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved full-mask scatter to %s", scatter_path)

    # ---- Comparison scatter: both on one plot ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["full_recall"], df["full_precision"], alpha=0.5, s=15,
               color="tomato", label=f"Full mirror  (mean P={mean_full_p:.3f})")
    ax.scatter(df["rcs_recall"],  df["rcs_precision"],  alpha=0.5, s=15,
               color="steelblue", label=f"RCS mask     (mean P={mean_rcs_p:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("Full Mirror Mask vs RCS Mask", fontsize=10)
    fig.tight_layout()
    comparison_path = output_dir / "full_vs_rcs_scatter.pdf"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison scatter to %s", comparison_path)


if __name__ == "__main__":
    main()
