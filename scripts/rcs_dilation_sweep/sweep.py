"""
RCS Dilation Radius Sweep Experiment
=====================================
For every Blender sample, runs MASt3R once using the horizontally-flipped
mirror view.

Correspondence points from that run form the combined mask. The dilation
radius is then swept on that mask and compared against the ground-truth
geometry constraint mask.

The best radius is selected by F₀.₅ (β=0.5): weights precision 4× over recall.

Usage
-----
    conda run -n fill-my-mirror python experiments/rcs_dilation_sweep/sweep.py \
        --config configs/config.yaml \
        --indices 0 1 2 \
        --radii 0 1 2 3 4 5 6 \
        --output-dir experiments/rcs_dilation_sweep/results
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fill_my_mirror.loaders import BlenderSampleLoader, GTGeometrySample
from fill_my_mirror.evaluation.rcs_mask_computation import (
    _ensure_mast3r,
    _run_mast3r_correspondences,
    _dilate_and_intersect,
    _pil_to_uint8_rgb,
    _pil_to_binary,
    _MAST3R_IMG_SIZE,
)
from fill_my_mirror.evaluation.constrained_pixels_gt_geometry_mask_computation import (
    compute_constrained_pixels_gt_geometry_mask,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_TRANSFORMS: list[str] = ["hflip"]

# Default F-beta parameter — β=0.1 is conservative: minimises false positives
_FBETA = 0.5


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _build_scene(image_arr: np.ndarray, mirror_mask_arr: np.ndarray) -> np.ndarray:
    scene = image_arr.copy()
    scene[mirror_mask_arr] = 0
    return scene


def _build_mirror(image_arr: np.ndarray, mirror_mask_arr: np.ndarray, transform: str) -> np.ndarray:
    mirror = image_arr.copy()
    mirror[~mirror_mask_arr] = 0
    if transform == "hflip":
        return np.fliplr(mirror)
    elif transform == "rot180":
        return np.rot90(mirror, 2)
    else:
        raise ValueError(f"Unknown transform: {transform!r}")


def _undo_base_transform(pts: np.ndarray, transform: str, S: int = _MAST3R_IMG_SIZE) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    if transform == "hflip":
        x = S - 1 - x
    elif transform == "rot180":
        x = S - 1 - x
        y = S - 1 - y
    return np.stack([x, y], axis=1)


def _mirror_pts_to_image(
    pts_mirror_raw: np.ndarray,
    transform: str,
    W: int,
    H: int,
    S: int = _MAST3R_IMG_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw MASt3R mirror pts to (xs, ys) in original image pixel coords."""
    pts = _undo_base_transform(pts_mirror_raw, transform, S)
    xs = np.clip(np.round(pts[:, 0] * (W / S)).astype(int), 0, W - 1)
    ys = np.clip(np.round(pts[:, 1] * (H / S)).astype(int), 0, H - 1)
    return xs, ys


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _fbeta(precision: float, recall: float, beta: float = _FBETA) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def _compute_pr(pred: np.ndarray, gt: np.ndarray, beta: float = _FBETA) -> tuple[float, float, float, float]:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fb        = _fbeta(precision, recall, beta)
    return precision, recall, f1, fb


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def _load_binary(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def _save_mask(arr: np.ndarray, path: Path) -> None:
    Image.fromarray((arr.astype(np.uint8) * 255), mode="L").save(path)


def _save_overlay(image_arr: np.ndarray, mask_arr: np.ndarray, path: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_arr)
    overlay = np.zeros((*mask_arr.shape, 4), dtype=np.float32)
    overlay[mask_arr] = [1.0, 0.0, 0.0, 0.5]
    ax.imshow(overlay)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def _evaluate_radii(
    sample_idx: int,
    image_arr: np.ndarray,
    mirror_mask: np.ndarray,
    combined_mask: np.ndarray,
    gt_mask: np.ndarray,
    radii: Sequence[int],
    iterations_list: Sequence[int],
    sample_dir: Path,
    beta: float = _FBETA,
) -> list[dict]:
    """Run dilation sweep and save per-(radius, iterations) outputs. Returns metric rows."""
    n_combined = int(combined_mask.sum())
    rows: list[dict] = []
    for r in radii:
        for iters in iterations_list:
            rcs_mask = _dilate_and_intersect(combined_mask, mirror_mask, r, iterations=iters)
            precision, recall, f1, fb = _compute_pr(rcs_mask, gt_mask, beta)

            tag = f"r{r}_i{iters}"
            _save_mask(rcs_mask, sample_dir / f"rcs_{tag}.png")
            _save_overlay(
                image_arr, rcs_mask, sample_dir / f"overlay_{tag}.png",
                title=f"r={r} iters={iters}  P={precision:.2f}  R={recall:.2f}  Fβ={fb:.2f}",
            )

            rows.append({
                "sample_idx":    sample_idx,
                "radius":        r,
                "iterations":    iters,
                "n_combined_px": n_combined,
                "precision":     precision,
                "recall":        recall,
                "f1":            f1,
                "fbeta":         fb,
            })
            logger.info("[%d] r=%3d iters=%d  P=%.3f  R=%.3f  F1=%.3f  Fβ=%.3f",
                        sample_idx, r, iters, precision, recall, f1, fb)
    return rows


def process_sample(
    sample_idx: int,
    sample: GTGeometrySample,
    radii: Sequence[int],
    iterations_list: Sequence[int],
    blender_path: Path,
    mast3r_model_name: str,
    device: str,
    output_dir: Path,
    beta: float = _FBETA,
) -> list[dict]:
    """Run MASt3R + dilation sweep for one sample. Returns metric rows."""
    sample_dir = output_dir / f"sample_{sample_idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Ground-truth mask
    # ------------------------------------------------------------------
    logger.info("[%d] Computing GT geometry constraint mask ...", sample_idx)
    gt_mask_path = compute_constrained_pixels_gt_geometry_mask(
        sample=sample,
        blender_path=blender_path,
        mask_stem=str(sample_idx),
    )
    gt_mask = _load_binary(gt_mask_path)

    image_arr   = _pil_to_uint8_rgb(Image.open(sample.gt_image_path))
    mirror_mask = _pil_to_binary(Image.open(sample.mask_path))
    H, W        = image_arr.shape[:2]

    _save_mask(gt_mask, sample_dir / "gt_mask.png")
    _save_overlay(image_arr, gt_mask, sample_dir / "overlay_gt.png", title="GT constraint mask")
    Image.fromarray(image_arr).save(sample_dir / "debug_image.png")

    scene_arr = _build_scene(image_arr, mirror_mask)

    # ------------------------------------------------------------------
    # 2. Run MASt3R with horizontal flip only
    # ------------------------------------------------------------------
    combined_mask = np.zeros((H, W), dtype=bool)

    for transform in _TRANSFORMS:
        logger.info("[%d] Transform: %s ...", sample_idx, transform)

        mirror_arr = _build_mirror(image_arr, mirror_mask, transform)

        vdir = sample_dir / transform
        vdir.mkdir(exist_ok=True)
        Image.fromarray(np.ascontiguousarray(scene_arr)).save(vdir / "view_scene.png")
        Image.fromarray(np.ascontiguousarray(mirror_arr)).save(vdir / "view_mirror.png")

        pts_scene_raw, pts_mirror_raw = _run_mast3r_correspondences(
            scene_arr, mirror_arr, mast3r_model_name, device,
        )

        n_pts = pts_mirror_raw.shape[0]
        logger.info("[%d] %s  correspondences: %d", sample_idx, transform, n_pts)

        correspondence_mask = np.zeros((H, W), dtype=bool)
        if n_pts > 0:
            xs, ys = _mirror_pts_to_image(pts_mirror_raw, transform, W, H)
            in_mirror = int(mirror_mask[ys, xs].sum())
            logger.info("[%d] %s  pts inside mirror: %d / %d",
                        sample_idx, transform, in_mirror, n_pts)
            correspondence_mask[ys, xs] = True

        _save_mask(correspondence_mask, vdir / "correspondence_mask.png")
        _save_overlay(image_arr, correspondence_mask, vdir / "overlay_correspondence.png",
                      title=f"{transform} raw corr")

        combined_mask |= correspondence_mask

    # ------------------------------------------------------------------
    # 3. Save the combined correspondence mask
    # ------------------------------------------------------------------
    n_combined = int(combined_mask.sum())
    logger.info("[%d] Correspondences (hflip): %d px", sample_idx, n_combined)
    _save_mask(combined_mask, sample_dir / "combined_correspondence_mask.png")
    _save_overlay(image_arr, combined_mask, sample_dir / "overlay_combined_correspondence.png",
                  title="Correspondences (hflip)")

    # ------------------------------------------------------------------
    # 4. Per-radius evaluation
    # ------------------------------------------------------------------
    return _evaluate_radii(sample_idx, image_arr, mirror_mask, combined_mask, gt_mask, radii, iterations_list, sample_dir, beta)


def redilate_sample(
    sample_idx: int,
    sample: GTGeometrySample,
    radii: Sequence[int],
    iterations_list: Sequence[int],
    output_dir: Path,
    beta: float = _FBETA,
) -> list[dict]:
    """Re-run dilation sweep using the saved correspondence mask. Returns metric rows."""
    sample_dir = output_dir / f"sample_{sample_idx}"

    combined_mask_path = sample_dir / "combined_correspondence_mask.png"
    gt_mask_path       = sample_dir / "gt_mask.png"

    if not combined_mask_path.exists():
        raise FileNotFoundError(
            f"[{sample_idx}] combined_correspondence_mask.png not found in {sample_dir}. "
            "Run without --redilate first."
        )
    if not gt_mask_path.exists():
        raise FileNotFoundError(
            f"[{sample_idx}] gt_mask.png not found in {sample_dir}. "
            "Run without --redilate first."
        )

    logger.info("[%d] Reusing saved correspondence mask from %s", sample_idx, sample_dir)
    combined_mask = _load_binary(combined_mask_path)
    gt_mask       = _load_binary(gt_mask_path)
    image_arr     = _pil_to_uint8_rgb(Image.open(sample.gt_image_path))
    mirror_mask   = _pil_to_binary(Image.open(sample.mask_path))

    return _evaluate_radii(sample_idx, image_arr, mirror_mask, combined_mask, gt_mask, radii, iterations_list, sample_dir, beta)


# ---------------------------------------------------------------------------
# Aggregate plots
# ---------------------------------------------------------------------------

def _plot_pr_curve(summary: pd.DataFrame, best_radius: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(summary["recall"], summary["precision"], "o-", color="steelblue")
    for _, row in summary.iterrows():
        ax.annotate(
            f"r={int(row['radius'])}",
            (row["recall"], row["precision"]),
            textcoords="offset points", xytext=(5, 5), fontsize=8,
        )
    best = summary[summary["radius"] == best_radius].iloc[0]
    ax.scatter([best["recall"]], [best["precision"]], color="red", zorder=5,
               label=f"best r={best_radius}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_vs_radius(summary: pd.DataFrame, metric: str, best_radius: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(summary["radius"], summary[metric], "o-", color="steelblue")
    best_val = summary.loc[summary["radius"] == best_radius, metric].values[0]
    ax.axvline(best_radius, color="red", linestyle="--", linewidth=1, label=f"best r={best_radius}")
    ax.scatter([best_radius], [best_val], color="red", zorder=5)
    ax.set_xlabel("Dilation radius")
    ax.set_ylabel(metric)
    # integer ticks for every radius value
    all_radii = sorted(summary["radius"].unique().tolist())
    ax.set_xticks(all_radii)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _write_summary_and_plots(metrics_df: pd.DataFrame, output_dir: Path, beta: float) -> None:
    """Aggregate metrics_df into summary.csv and regenerate all plots."""
    summary = (
        metrics_df
        .groupby(["radius", "iterations"])[["precision", "recall", "f1", "fbeta"]]
        .mean()
        .reset_index()
    )

    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved summary to %s", summary_path)

    best_idx    = summary["fbeta"].idxmax()
    best_radius = int(summary.loc[best_idx, "radius"])
    best_iters  = int(summary.loc[best_idx, "iterations"])
    best        = summary.loc[best_idx]

    for iters in summary["iterations"].unique():
        sub = summary[summary["iterations"] == iters].copy()
        tag = f"_i{iters}"
        br  = int(sub.loc[sub["fbeta"].idxmax(), "radius"])
        for stem in [f"precision_recall{tag}", f"precision_vs_radius{tag}",
                     f"recall_vs_radius{tag}", f"f1_vs_radius{tag}", f"fbeta_vs_radius{tag}"]:
            old_png = output_dir / f"{stem}.png"
            if old_png.exists():
                old_png.unlink()
        _plot_pr_curve(sub, br, output_dir / f"precision_recall{tag}.pdf")
        _plot_metric_vs_radius(sub, "precision", br, output_dir / f"precision_vs_radius{tag}.pdf")
        _plot_metric_vs_radius(sub, "recall",    br, output_dir / f"recall_vs_radius{tag}.pdf")
        _plot_metric_vs_radius(sub, "f1",        br, output_dir / f"f1_vs_radius{tag}.pdf")
        _plot_metric_vs_radius(sub, "fbeta",     br, output_dir / f"fbeta_vs_radius{tag}.pdf")

    print("\n" + "=" * 60)
    print("Summary (mean across samples):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 60)
    print(
        f"\nSuggested dilation_radius = {best_radius}, iterations = {best_iters}  "
        f"(F{beta}={best['fbeta']:.4f}, "
        f"P={best['precision']:.4f}, "
        f"R={best['recall']:.4f})"
    )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep dilation radius for RCS mask tuning (horizontal flip only)."
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--indices", type=int, nargs="*", default=None,
        help="Subset of Blender dataset indices to process. Default: all.",
    )
    parser.add_argument(
        "--radii", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6],
        help="Dilation radii to sweep.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/rcs_dilation_sweep/results",
        help="Directory for all output files.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device (cuda/cpu).")
    parser.add_argument(
        "--redilate", action="store_true",
        help=(
            "Skip MASt3R inference and reuse the saved combined_correspondence_mask.png "
            "from a previous run. Only re-runs dilation for the given --radii/--iterations "
            "and overwrites all per-radius masks, overlays, and summary outputs."
        ),
    )
    parser.add_argument(
        "--iterations", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7],
        help="Dilation iteration counts to sweep (applied per radius). Default: 2.",
    )
    parser.add_argument(
        "--beta", type=float, default=_FBETA,
        help=(
            f"F-beta parameter for selecting the best (radius, iterations) pair. "
            f"Smaller β weights precision more heavily (default: {_FBETA}). "
            "β=0.1 is very conservative (minimises false positives)."
        ),
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help=(
            "Skip all dilation computation. Load the existing metrics.csv from --output-dir, "
            "recompute fbeta with --beta, and regenerate summary.csv and all plots. "
            "Use this to try a different --beta without re-running dilation."
        ),
    )
    args = parser.parse_args()

    beta = args.beta

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Summary-only mode: recompute fbeta from existing metrics.csv
    # ------------------------------------------------------------------
    if args.summary_only:
        metrics_path = output_dir / "metrics.csv"
        if not metrics_path.exists():
            logger.error("metrics.csv not found in %s. Run the full sweep first.", output_dir)
            sys.exit(1)
        logger.info("Summary-only mode: loading %s and recomputing fbeta with β=%.3f", metrics_path, beta)
        metrics_df = pd.read_csv(metrics_path)
        metrics_df["fbeta"] = metrics_df.apply(
            lambda row: _fbeta(row["precision"], row["recall"], beta), axis=1
        )
        _write_summary_and_plots(metrics_df, output_dir, beta)
        return

    with open(args.config) as f:
        config = yaml.safe_load(f)

    loader  = BlenderSampleLoader()
    indices = args.indices if args.indices is not None else list(range(len(loader)))

    if args.redilate:
        logger.info(
            "Re-dilate mode: reusing saved correspondence masks for %d samples × %d radii × %d iteration counts",
            len(indices), len(args.radii), len(args.iterations),
        )
        all_rows: list[dict] = []
        for idx in indices:
            if not (0 <= idx < len(loader)):
                logger.warning("Index %d out of range [0, %d), skipping.", idx, len(loader))
                continue
            sample = loader.load(idx)
            try:
                rows = redilate_sample(
                    sample_idx=idx,
                    sample=sample,
                    radii=args.radii,
                    iterations_list=args.iterations,
                    output_dir=output_dir,
                    beta=beta,
                )
            except FileNotFoundError as e:
                logger.error("%s", e)
                sys.exit(1)
            all_rows.extend(rows)
    else:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        if not _ensure_mast3r():
            logger.error("MASt3R not available. Aborting.")
            sys.exit(1)

        blender_path = Path(config["blender_path"])
        if not blender_path.exists():
            logger.error("Blender not found at %s", blender_path)
            sys.exit(1)

        mast3r_model_name = config.get(
            "mast3r_model_name", "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        )

        logger.info(
            "Processing %d samples (hflip only) × %d radii × %d iteration counts",
            len(indices), len(args.radii), len(args.iterations),
        )

        all_rows = []
        for idx in indices:
            if not (0 <= idx < len(loader)):
                logger.warning("Index %d out of range [0, %d), skipping.", idx, len(loader))
                continue
            sample = loader.load(idx)
            rows = process_sample(
                sample_idx=idx,
                sample=sample,
                radii=args.radii,
                iterations_list=args.iterations,
                blender_path=blender_path,
                mast3r_model_name=mast3r_model_name,
                device=device,
                output_dir=output_dir,
                beta=beta,
            )
            all_rows.extend(rows)

    if not all_rows:
        logger.error("No results produced.")
        sys.exit(1)

    metrics_df = pd.DataFrame(all_rows)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved per-sample metrics to %s", metrics_path)

    _write_summary_and_plots(metrics_df, output_dir, beta)


if __name__ == "__main__":
    main()
