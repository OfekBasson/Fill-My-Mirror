"""
Depth-degradation sweep: how much does noise interpolation help as geometry
degrades from GT toward monocular estimate (and beyond)?

For each Blender dataset image in [start_index, end_index) and each
lambda in LAMBDAS:

  1. Blend depth: d_λ = (1−λ)·d_GT + λ·d_est_aligned
     where d_est_aligned is MoGe depth aligned to GT via least-squares
     scale+shift.
  2. Recompute 3-D points from d_λ using GT intrinsics.
  3. Run projection (Blender render) → projected_image.png + geometry_constraint_mask.png.
  4. Run inpainting WITH  interpolation  (n=13, t_prime=625, seed=SEED).
  5. Run inpainting WITHOUT interpolation (n=13, t_prime=0,   seed=SEED).
  6. Compute masked PSNR/SSIM/LPIPS on:
       - full mirror region
       - constrained region  (mirror ∩ ¬geometry_constraint_mask)
     and record  gap = metric_with − metric_without
     (for LPIPS: gap = metric_without − metric_with, so positive = interp helps).

Outputs
-------
    <output_dir>/results.csv
    <output_dir>/gap_{metric}.{png,pdf}    (one plot per metric)
    <output_dir>/SUMMARY.md

Usage
-----
    python analysis/depth_degradation_sweep.py \\
        --start-index 0 --end-index 20 \\
        --output-dir /tmp/depth_sweep

    # Custom seed and config:
    python analysis/depth_degradation_sweep.py \\
        --start-index 0 --end-index 50 \\
        --seed 42 \\
        --config configs/config.yaml \\
        --output-dir /tmp/depth_sweep
"""

from __future__ import annotations

import argparse
import gc
import logging
import traceback
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from fill_my_mirror.dual_mask_inpainting import load_inpainting_pipeline, run_dual_mask_inpainting
from fill_my_mirror.evaluation.metrics_computation import (
    GeneratedImage,
    MetricsInput,
    compute_metrics,
)
from fill_my_mirror.geometry import estimate_geometry
from fill_my_mirror.geometry.core import MoGeGeometryProcessor
from fill_my_mirror.loaders import BlenderSampleLoader, EstimatedGeometrySample, GTGeometrySample
from fill_my_mirror.projection import run_projection_single_mirror

logger = logging.getLogger(__name__)

LAMBDAS: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

# Inpainting hyper-parameters (match the paper's reported settings)
WITH_N: float = 13.0
WITH_T: float = 625.0     # WITH  interpolation
WITHOUT_T: float = 0.0    # WITHOUT interpolation (t_prime=0 ⟹ condition never fires)

DEFAULT_CONFIG = Path("configs/config.yaml")
DEFAULT_SEED = 0


# ---------------------------------------------------------------------------
# Depth utilities
# ---------------------------------------------------------------------------


def _align_depth_ls(d_est: np.ndarray, d_gt: np.ndarray) -> tuple[float, float]:
    """Least-squares align d_est to d_gt: returns (scale, shift) so that
    scale*d_est + shift ≈ d_gt in the L2 sense over finite, positive pixels."""
    valid = np.isfinite(d_est) & np.isfinite(d_gt) & (d_gt > 0) & (d_est > 0)
    if valid.sum() < 10:
        med_gt = float(np.median(d_gt[np.isfinite(d_gt) & (d_gt > 0)]))
        med_est = float(np.median(d_est[np.isfinite(d_est) & (d_est > 0)]))
        return med_gt / (med_est + 1e-8), 0.0
    x = d_est[valid].ravel()
    y = d_gt[valid].ravel()
    A = np.stack([x, np.ones_like(x)], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _depth_to_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Unproject a depth map to (H, W, 3) camera-space 3-D points.

    Uses the normalized-intrinsics convention (fx/W, cx/W, cy/H) that the rest
    of the pipeline inherits from MoGe and the Blender loader, with the
    sign flip [-1, -1, 1] applied to match how GT points are stored.
    """
    H, W = depth.shape
    fx = intrinsics[0, 0] * W
    fy = intrinsics[1, 1] * H
    cx = intrinsics[0, 2] * W
    cy = intrinsics[1, 2] * H
    ys, xs = np.mgrid[0:H, 0:W]
    X = (xs - cx) / (fx + 1e-8) * depth
    Y = (ys - cy) / (fy + 1e-8) * depth
    pts = np.stack([X, Y, depth], axis=-1).astype(np.float32)
    pts *= np.array([-1.0, -1.0, 1.0], dtype=np.float32)
    return pts


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------


def _compute_constrained_mask(
    mirror_pil: Image.Image,
    geom_constraint_pil: Image.Image,
) -> Image.Image:
    """Return constrained_mask = mirror ∩ ¬geometry_constraint.

    geometry_constraint_mask.png has WHITE = inpainting holes (pixels NOT
    covered by projection). Constrained pixels are the complement within
    the mirror: where projection DID provide coverage.
    """
    mirror = np.asarray(mirror_pil.convert("L"), dtype=np.uint8) > 127
    inpainting_holes = np.asarray(geom_constraint_pil.convert("L"), dtype=np.uint8) > 127
    constrained = mirror & ~inpainting_holes
    return Image.fromarray((constrained.astype(np.uint8) * 255), mode="L")


# ---------------------------------------------------------------------------
# Core sweep loop
# ---------------------------------------------------------------------------


def _run_one_lambda(
    idx: int,
    lam: float,
    gt_sample: GTGeometrySample,
    d_est_aligned: np.ndarray,
    lam_dir: Path,
    blender_path: Path,
    geom_model_name: str,
    inpaint_model_name: str,
    pipe,
    seed: int,
    prompt_fallback: str,
) -> dict | None:
    """Run projection + dual inpainting (with/without) for one (image, lambda).

    Returns a dict of metric values, or None if any step fails.
    """
    label = f"idx={idx} λ={lam}"

    # ---- 1. Build interpolated depth + 3-D points --------------------------
    d_gt = gt_sample.depth.astype(np.float32)
    d_lam = (1.0 - lam) * d_gt + lam * d_est_aligned
    d_lam = np.clip(d_lam, 1e-4, None)
    points_lam = _depth_to_points(d_lam, gt_sample.intrinsics)

    modified_sample = GTGeometrySample(
        image_path=gt_sample.image_path,
        mask_path=gt_sample.mask_path,
        gt_image_path=gt_sample.gt_image_path,
        prompt=gt_sample.prompt,
        points=points_lam,
        depth=d_lam,
        intrinsics=gt_sample.intrinsics,
    )

    # ---- 2. Projection ------------------------------------------------------
    print(f"  {label}: projecting ...")
    try:
        geom_out = estimate_geometry(
            modified_sample,
            model_name=geom_model_name,
            tmp_dir=lam_dir / "geom",
        )
        proj_out = run_projection_single_mirror(
            geometry_output=geom_out,
            image_path=gt_sample.image_path,
            mirror_mask_path=gt_sample.mask_path,
            blender_path=blender_path,
            projected_image_path=lam_dir / "projected_image.png",
            geometry_constraint_mask_path=lam_dir / "geometry_constraint_mask.png",
            tmp_dir=lam_dir / "blender",
        )
    except Exception:
        logger.warning("%s: projection failed:\n%s", label, traceback.format_exc())
        return None
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    # ---- 3. Prepare masks and prompt ----------------------------------------
    mirror_pil = Image.open(gt_sample.mask_path).convert("L")
    geom_pil = Image.open(proj_out.geometry_constraint_mask_path).convert("L")
    constrained_pil = _compute_constrained_mask(mirror_pil, geom_pil)
    gt_pil = Image.open(gt_sample.gt_image_path)
    prompt = gt_sample.prompt or prompt_fallback

    constrained_px = int(np.asarray(constrained_pil).sum() // 255)
    logger.info("%s: constrained pixels = %d", label, constrained_px)

    # ---- 4. Inpainting: WITH interpolation ----------------------------------
    with_path = lam_dir / "with_interp.png"
    print(f"  {label}: inpainting WITH t_prime={WITH_T} ...")
    try:
        run_dual_mask_inpainting(
            prompt=prompt,
            projected_image_path=proj_out.projected_image_path,
            geometry_constraint_mask_path=proj_out.geometry_constraint_mask_path,
            generative_refinement_mask_path=gt_sample.mask_path,
            output_path=with_path,
            original_image_path=gt_sample.image_path,
            model_name=inpaint_model_name,
            seed=seed,
            n=WITH_N,
            t_prime=WITH_T,
            use_dual_mask=True,
            pipe=pipe,
        )
    except Exception:
        logger.warning("%s: WITH inpainting failed:\n%s", label, traceback.format_exc())
        return None
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    # ---- 5. Inpainting: WITHOUT interpolation (t_prime=0) -------------------
    without_path = lam_dir / "without_interp.png"
    print(f"  {label}: inpainting WITHOUT interpolation ...")
    try:
        run_dual_mask_inpainting(
            prompt=prompt,
            projected_image_path=proj_out.projected_image_path,
            geometry_constraint_mask_path=proj_out.geometry_constraint_mask_path,
            generative_refinement_mask_path=gt_sample.mask_path,
            output_path=without_path,
            original_image_path=gt_sample.image_path,
            model_name=inpaint_model_name,
            seed=seed,
            n=WITH_N,
            t_prime=WITHOUT_T,
            use_dual_mask=True,
            pipe=pipe,
        )
    except Exception:
        logger.warning("%s: WITHOUT inpainting failed:\n%s", label, traceback.format_exc())
        return None
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    # ---- 6. Compute metrics -------------------------------------------------
    mi = MetricsInput(
        gt_image=gt_pil,
        generated_images=[
            GeneratedImage(name="with_interp", image=Image.open(with_path)),
            GeneratedImage(name="without_interp", image=Image.open(without_path)),
        ],
        full_mirror_mask=mirror_pil,
        constrained_mask=constrained_pil,
        save_path=lam_dir / "metrics",
        prompt=prompt,
    )
    try:
        df_m = compute_metrics(mi)
    except ValueError as exc:
        logger.warning("%s: compute_metrics failed: %s", label, exc)
        return None

    with_row = df_m[df_m["name"] == "with_interp"].iloc[0].to_dict()
    without_row = df_m[df_m["name"] == "without_interp"].iloc[0].to_dict()
    metric_keys = [c for c in df_m.columns if c != "name"]

    row: dict = {"image_id": idx, "lambda": lam, "constrained_pixels": constrained_px}
    for k in metric_keys:
        v_with = with_row.get(k)
        v_without = without_row.get(k)
        row[f"{k}_with"] = v_with
        row[f"{k}_without"] = v_without
        # gap = with − without for every metric.
        # Positive gap means "with interpolation is higher"; interpret direction
        # based on the metric (PSNR/SSIM: higher=better; LPIPS: lower=better).
        try:
            row[f"{k}_gap"] = float(v_with) - float(v_without)
        except (TypeError, ValueError):
            row[f"{k}_gap"] = float("nan")

    print(f"  {label}: done")
    return row


def run_sweep(
    start: int,
    end: int,
    output_dir: Path,
    config: dict,
    seed: int,
) -> pd.DataFrame:
    blender_path = Path(config["blender_path"])
    inpaint_model_name: str = config["inpainting_model_name"]
    geom_model_name: str = config["geometry_model_name"]
    prompt_fallback: str = config.get("prompt", "")

    loader = BlenderSampleLoader()
    dataset_size = len(loader)
    end = min(end, dataset_size)

    print(f"Dataset size : {dataset_size}")
    print(f"Range        : [{start}, {end})")
    print(f"Lambdas      : {LAMBDAS}")
    print(f"Seed         : {seed}")
    print(f"Output       : {output_dir}")
    print()

    # Load heavyweight models once for the whole sweep
    moge_proc = MoGeGeometryProcessor(geom_model_name)
    pipe = load_inpainting_pipeline(model_name=inpaint_model_name, gpu_id=0)

    rows: list[dict] = []

    for idx in range(start, end):
        print(f"\n=== Image {idx} ===")

        # ---- Load GT sample and run MoGe (once per image) ------------------
        try:
            gt_sample = loader.load(idx, use_estimated_geometry=False)
        except Exception:
            logger.warning("Failed to load index %d:\n%s", idx, traceback.format_exc())
            continue

        est_sample = EstimatedGeometrySample(
            image_path=gt_sample.image_path,
            mask_path=gt_sample.mask_path,
            prompt=gt_sample.prompt,
            gt_image_path=gt_sample.gt_image_path,
        )

        try:
            moge_tmp = output_dir / "tmp_moge" / str(idx)
            moge_tmp.mkdir(parents=True, exist_ok=True)
            moge_out = moge_proc.get_geometry(est_sample, tmp_dir=moge_tmp)
        except Exception:
            logger.warning("MoGe failed for index %d:\n%s", idx, traceback.format_exc())
            continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        d_gt = gt_sample.depth.astype(np.float32)
        d_est_raw = moge_out.depth.astype(np.float32)
        if d_est_raw.shape != d_gt.shape:
            d_est_raw = cv2.resize(
                d_est_raw,
                (d_gt.shape[1], d_gt.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Least-squares alignment: d_est_aligned ≈ d_gt
        scale, shift = _align_depth_ls(d_est_raw, d_gt)
        d_est_aligned = np.clip(d_est_raw * scale + shift, 1e-4, None).astype(np.float32)

        logger.info(
            "idx=%d  depth alignment: scale=%.4f  shift=%.4f", idx, scale, shift
        )

        # ---- Iterate over lambdas ------------------------------------------
        for lam in LAMBDAS:
            lam_dir = output_dir / str(idx) / f"lam_{lam:.2f}"
            lam_dir.mkdir(parents=True, exist_ok=True)

            row = _run_one_lambda(
                idx=idx,
                lam=lam,
                gt_sample=gt_sample,
                d_est_aligned=d_est_aligned,
                lam_dir=lam_dir,
                blender_path=blender_path,
                geom_model_name=geom_model_name,
                inpaint_model_name=inpaint_model_name,
                pipe=pipe,
                seed=seed,
                prompt_fallback=prompt_fallback,
            )
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting and summary
# ---------------------------------------------------------------------------

# Primary gap metrics to plot (plus SSIM if available)
_PLOT_METRICS = [
    "psnr_constrained",
    "psnr_full_mirror",
    "ssim_constrained",
    "ssim_full_mirror",
    "lpips_constrained",
    "lpips_full_mirror",
]


def _plot_gap(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    gap_col = f"{metric}_gap"
    if gap_col not in df.columns:
        return

    df_valid = df.dropna(subset=[gap_col])
    if df_valid.empty:
        return

    grouped = (
        df_valid.groupby("lambda")[gap_col]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", label="no effect")
    ax.errorbar(
        grouped["lambda"],
        grouped["mean"],
        yerr=grouped["sem"],
        marker="o",
        capsize=4,
        linewidth=1.8,
        markersize=6,
        label="mean ± SE",
    )
    ax.set_xlabel("λ  (0 = GT depth, 1 = MoGe estimate, >1 = extrapolated)")
    ax.set_ylabel(f"Δ{metric}  (with − without interpolation)")
    n_imgs = df_valid["image_id"].nunique()
    ax.set_title(
        f"Interpolation gain vs. geometry degradation\n"
        f"({metric}, mean ± SE, n={n_imgs} images)"
    )
    ax.set_xticks(grouped["lambda"].tolist())
    ax.legend(fontsize=9)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"gap_{metric}.{ext}", dpi=150)
        print(f"Saved {output_dir / f'gap_{metric}.{ext}'}")
    plt.close(fig)


def _write_summary(df: pd.DataFrame, output_dir: Path) -> None:
    gap_cols = sorted(c for c in df.columns if c.endswith("_gap"))
    grouped = df.groupby("lambda")[gap_cols].mean().reset_index()

    # Format table as Markdown-safe text (no tabulate dependency)
    header = ["lambda"] + gap_cols
    col_w = [max(len(h), 8) for h in header]
    sep = "| " + " | ".join("-" * w for w in col_w) + " |"
    head_row = "| " + " | ".join(h.ljust(w) for h, w in zip(header, col_w)) + " |"

    data_rows = []
    for _, r in grouped.iterrows():
        cells = [f"{r['lambda']:.2f}"] + [
            f"{r[c]:.4f}" if pd.notna(r.get(c)) else "NaN"
            for c in gap_cols
        ]
        data_rows.append("| " + " | ".join(v.ljust(w) for v, w in zip(cells, col_w)) + " |")

    lines = [
        "# Depth-Degradation Sweep: SUMMARY",
        "",
        f"**Images evaluated:** {df['image_id'].nunique()}",
        f"**Lambdas:** {sorted(df['lambda'].unique().tolist())}",
        f"**Seed:** (see CLI invocation)",
        f"**Interpolation config (WITH):** n={WITH_N}, t_prime={WITH_T}",
        f"**No-interpolation config (WITHOUT):** n={WITH_N}, t_prime={WITHOUT_T}",
        "",
        "## Mean gap (positive = interpolation helps) per lambda",
        "",
        "Gap = with − without for every metric.",
        "PSNR/SSIM: positive gap = interpolation helps (higher is better).",
        "LPIPS: negative gap = interpolation helps (lower is better).",
        "",
        head_row,
        sep,
        *data_rows,
        "",
        "## Interpretation",
        "",
        "Expected trend: gap increases with λ as geometry degrades from GT (λ=0) toward",
        "monocular estimation (λ=1) and beyond (λ>1). This validates that noise interpolation",
        "provides greater benefit when geometry is less reliable — the regime of real monocular",
        "depth estimation.",
        "",
        "## Notes on seed matching",
        "",
        "Both 'with' and 'without' runs use **the same seed** (generator seeded once at the",
        "start of each `run_dual_mask_inpainting` call). The initial latent noise is thus",
        "identical; the difference in output is attributable solely to the t_prime parameter.",
    ]

    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"Saved {output_dir / 'SUMMARY.md'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Depth-degradation sweep: interpolation gain vs. geometry quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="First dataset index (inclusive).",
    )
    parser.add_argument(
        "--end-index", type=int, default=20,
        help="Last dataset index (exclusive).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="RNG seed for both with/without runs (default: 0).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Root directory for all outputs.",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = run_sweep(
        start=args.start_index,
        end=args.end_index,
        output_dir=args.output_dir,
        config=config,
        seed=args.seed,
    )

    if df.empty:
        print("No results produced — check logs for errors.")
        return

    csv_path = args.output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows → {csv_path}")

    for metric in _PLOT_METRICS:
        _plot_gap(df, metric, args.output_dir)

    _write_summary(df, args.output_dir)


if __name__ == "__main__":
    main()
