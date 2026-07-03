"""
Depth-degradation sweep: how much does noise interpolation help as geometry
degrades from GT toward monocular estimate (and beyond)?

For each MirrorBench-V2 dataset image in [start_index, end_index) and each
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

Outputs (local and uploaded to R2 under experiments/depth_degradation_sweep/<run_name>/)
---------------------------------------------------------------------------------------
    <output_dir>/results.csv
    <output_dir>/gap_{metric}.{png,pdf}    (one plot per metric)
    <output_dir>/SUMMARY.md
    <output_dir>/results_worker_<gpu_id>.csv   (partial, one per GPU)

Usage
-----
    python scripts/depth_degradation_sweep.py --num-samples 20 --output-dir /tmp/depth_sweep

    # Multi-GPU, custom seed and run name:
    python scripts/depth_degradation_sweep.py \\
        --num-samples 50 \\
        --seed 42 \\
        --num-gpus 2 \\
        --run-name my_run \\
        --output-dir /tmp/depth_sweep

--num-samples counts *successfully evaluated* images: indices are consumed
starting from --start-index and images that fail to load or get abandoned
(e.g. LowFiniteMirrorPointsRatioError) don't count against the total and are
simply skipped over.
"""

from __future__ import annotations

import argparse
import gc
import logging
import multiprocessing
import os
import time
import traceback
from pathlib import Path

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
from fill_my_mirror.geometry.core import LowFiniteMirrorPointsRatioError, MoGeDepthDegradationProcessor
from fill_my_mirror.loaders import MirrorBenchV2SampleLoader, DepthDegradedSample
from fill_my_mirror.projection import run_projection_single_mirror
from fill_my_mirror.storage import R2Client

logger = logging.getLogger(__name__)

LAMBDAS: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

# Inpainting hyper-parameters (match the paper's reported settings)
WITH_N: float = 13.0
WITH_T: float = 625.0     # WITH  interpolation
WITHOUT_T: float = 0.0    # WITHOUT interpolation (t_prime=0 ⟹ condition never fires)

DEFAULT_CONFIG = Path("configs/config.yaml")
DEFAULT_SEED = 0
DEFAULT_UPLOAD_EVERY = 5  # flush R2 uploads after this many images

R2_EXPERIMENT_PREFIX = "experiments/depth_degradation_sweep"


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------


def _worker(
    gpu_id: int,
    indices: list[int],
    target_successes: int,
    args_dict: dict,
    config: dict,
) -> None:
    # Restrict this process to a single physical GPU so both MoGe (inside
    # MoGeDepthDegradationProcessor) and the inpainting pipeline land on it.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    output_dir = Path(args_dict["output_dir"])
    seed = args_dict["seed"]
    run_name = args_dict["run_name"]
    upload_every = args_dict["upload_every"]
    skip_existing = args_dict["skip_existing"]

    blender_path = Path(config["blender_path"])
    inpaint_model_name: str = config["inpainting_model_name"]
    geom_model_name: str = config["geometry_model_name"]
    prompt_fallback: str = config.get("prompt", "")

    r2_prefix = f"{R2_EXPERIMENT_PREFIX}/{run_name}"
    r2 = R2Client()

    # Load any partial results from a previous interrupted run so that
    # skip_existing can avoid re-processing already-finished images.
    partial_csv = output_dir / f"results_worker_{gpu_id}.csv"
    if partial_csv.exists():
        existing_df = pd.read_csv(partial_csv)
        rows: list[dict] = existing_df.to_dict("records")
        already_done: set[int] = set(existing_df["image_id"].unique().tolist())
    else:
        rows = []
        already_done = set()

    num_successes = len(already_done)

    # gpu_id=0 because CUDA_VISIBLE_DEVICES already restricts to one device.
    # MoGeDepthDegradationProcessor loads MoGe once and caches per-image depth alignment.
    degradation_proc = MoGeDepthDegradationProcessor(geom_model_name)
    pipe = load_inpainting_pipeline(model_name=inpaint_model_name, gpu_id=0)

    loader = MirrorBenchV2SampleLoader()

    pending_upload: list[tuple[Path, str]] = []  # (local_dir, r2_key_prefix)

    def flush() -> None:
        for local_dir, r2_key in pending_upload:
            r2.upload_dir(local_dir, r2_key)
        pending_upload.clear()

    for idx in indices:
        print(f"\n[GPU {gpu_id}] === Image {idx} ===")

        if skip_existing and idx in already_done:
            print(f"[GPU {gpu_id}] Skipping {idx} (already in partial CSV)")
            continue

        try:
            gt_sample = loader.load(idx, use_estimated_geometry=False)
        except Exception:
            logger.warning("[GPU %d] Failed to load index %d:\n%s", gpu_id, idx, traceback.format_exc())
            continue

        prompt = gt_sample.prompt or prompt_fallback
        mirror_pil = Image.open(gt_sample.mask_path).convert("L")
        gt_pil = Image.open(gt_sample.gt_image_path)

        # Download the GT-geometry constrained mask once per image.
        # This mask (mirror ∩ ¬geometry_constraint at λ=0) is fixed across all λ
        # so metrics are always evaluated on the same region.
        gt_constrained_mask_path = output_dir / str(idx) / "constrained_pixels_gt_geometry_mask.png"
        gt_constrained_mask_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            r2.download_file(f"mirrorbench_v2/gt_geometry/{idx}/constrained_pixels_gt_geometry_mask.png", gt_constrained_mask_path)
            constrained_pil = Image.open(gt_constrained_mask_path).convert("L")
        except Exception:
            logger.warning("[GPU %d] idx=%d: GT constrained mask missing in R2, skipping image", gpu_id, idx)
            continue

        constrained_px = int(np.asarray(constrained_pil).sum() // 255)

        # ---- Iterate over lambdas ------------------------------------------
        # MoGe runs on the first lambda and is cached for the rest.
        image_abandoned = False
        for lam in LAMBDAS:
            label = f"[GPU {gpu_id}] idx={idx} λ={lam}"
            lam_dir = output_dir / str(idx) / f"lam_{lam:.2f}"
            lam_dir.mkdir(parents=True, exist_ok=True)

            # ---- Projection -------------------------------------------------
            print(f"  {label}: projecting ...")
            try:
                degraded_sample = DepthDegradedSample(
                    image_path=gt_sample.image_path,
                    mask_path=gt_sample.mask_path,
                    gt_image_path=gt_sample.gt_image_path,
                    prompt=gt_sample.prompt,
                    points=gt_sample.points,
                    depth=gt_sample.depth,
                    intrinsics=gt_sample.intrinsics,
                    lam=lam,
                    image_id=idx,
                )
                geom_out = degradation_proc.get_geometry(degraded_sample, tmp_dir=lam_dir / "geom")
                proj_out = run_projection_single_mirror(
                    geometry_output=geom_out,
                    image_path=gt_sample.image_path,
                    mirror_mask_path=gt_sample.mask_path,
                    blender_path=blender_path,
                    projected_image_path=lam_dir / "projected_image.png",
                    geometry_constraint_mask_path=lam_dir / "geometry_constraint_mask.png",
                    tmp_dir=lam_dir / "blender",
                )
            except LowFiniteMirrorPointsRatioError:
                logger.warning(
                    "%s: too few finite mirror points, abandoning image %d:\n%s",
                    label, idx, traceback.format_exc(),
                )
                image_abandoned = True
                break
            except Exception:
                logger.warning("%s: projection failed:\n%s", label, traceback.format_exc())
                continue
            finally:
                gc.collect()
                torch.cuda.empty_cache()

            # ---- Inpainting: WITH interpolation -----------------------------
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
                continue
            finally:
                gc.collect()
                torch.cuda.empty_cache()

            # ---- Inpainting: WITHOUT interpolation (t_prime=0) --------------
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
                continue
            finally:
                gc.collect()
                torch.cuda.empty_cache()

            # ---- Metrics ----------------------------------------------------
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
                continue

            with_row = df_m[df_m["name"] == "with_interp"].iloc[0].to_dict()
            without_row = df_m[df_m["name"] == "without_interp"].iloc[0].to_dict()
            metric_keys = [c for c in df_m.columns if c != "name"]

            row: dict = {"image_id": idx, "lambda": lam, "constrained_pixels": constrained_px}
            for k in metric_keys:
                v_with = with_row.get(k)
                v_without = without_row.get(k)
                row[f"{k}_with"] = v_with
                row[f"{k}_without"] = v_without
                try:
                    row[f"{k}_gap"] = float(v_with) - float(v_without)
                except (TypeError, ValueError):
                    row[f"{k}_gap"] = float("nan")

            print(f"  {label}: done")
            rows.append(row)

        if image_abandoned:
            continue

        # Queue this image's output directory for upload
        idx_dir = output_dir / str(idx)
        pending_upload.append((idx_dir, f"{r2_prefix}/{idx}"))

        if len(pending_upload) >= upload_every:
            flush()

        if any(r["image_id"] == idx for r in rows):
            num_successes += 1
            if target_successes is not None and num_successes >= target_successes:
                print(f"[GPU {gpu_id}] Reached target of {target_successes} successful images, stopping.")
                break

    flush()

    # Persist partial results and upload the CSV
    if rows:
        pd.DataFrame(rows).to_csv(partial_csv, index=False)
        r2.upload_file(partial_csv, f"{r2_prefix}/results_worker_{gpu_id}.csv")

    print(f"[GPU {gpu_id}] done.")


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
        help="First dataset index to consider (inclusive).",
    )
    parser.add_argument(
        "--num-samples", type=int, default=20,
        help="Number of images to successfully evaluate. Indices are consumed "
             "starting from --start-index until this many images produce at "
             "least one usable result row (images that fail to load, or that "
             "are abandoned due to LowFiniteMirrorPointsRatioError, don't count "
             "and are skipped over) or the dataset is exhausted.",
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
    parser.add_argument(
        "--num-gpus", type=int, default=None,
        help="Number of GPUs to use (default: all available).",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Identifier for this run, used as R2 subdirectory name. "
             "Defaults to a timestamp (sweep_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--upload-every", type=int, default=DEFAULT_UPLOAD_EVERY,
        help="Upload to R2 after this many images (per worker).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip images that already appear in the worker's partial CSV.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.run_name is None:
        args.run_name = f"sweep_{time.strftime('%Y%m%d_%H%M%S')}"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    loader = MirrorBenchV2SampleLoader()
    dataset_size = len(loader)
    # All indices from start_index to the end of the dataset are made available
    # to workers; each worker stops early once it hits its target number of
    # successful images, so --num-samples always means "N successful images"
    # rather than "N attempts" (some indices may fail to load or get abandoned).
    indices = list(range(args.start_index, dataset_size))

    num_gpus = args.num_gpus or torch.cuda.device_count() or 1

    r2_prefix = f"{R2_EXPERIMENT_PREFIX}/{args.run_name}"

    print(f"Run name     : {args.run_name}")
    print(f"Dataset size : {dataset_size}")
    print(f"Start index  : {args.start_index}")
    print(f"Num samples  : {args.num_samples}")
    print(f"Lambdas      : {LAMBDAS}")
    print(f"Seed         : {args.seed}")
    print(f"GPUs         : {num_gpus}")
    print(f"Output       : {args.output_dir}")
    print(f"R2 prefix    : {r2_prefix}")
    print()

    args_dict = {
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "run_name": args.run_name,
        "upload_every": args.upload_every,
        "skip_existing": args.skip_existing,
    }

    if num_gpus == 1:
        _worker(0, indices, args.num_samples, args_dict, config)
    else:
        chunk = (len(indices) + num_gpus - 1) // num_gpus
        chunks = [indices[i * chunk:(i + 1) * chunk] for i in range(num_gpus)]
        per_worker_target = (args.num_samples + num_gpus - 1) // num_gpus
        processes = []
        for gpu_id, chunk_indices in enumerate(chunks):
            if not chunk_indices:
                continue
            p = multiprocessing.Process(
                target=_worker,
                args=(gpu_id, chunk_indices, per_worker_target, args_dict, config),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    # ---- Merge per-worker CSVs and produce final outputs -------------------
    dfs = []
    for gpu_id in range(num_gpus):
        partial = args.output_dir / f"results_worker_{gpu_id}.csv"
        if partial.exists():
            dfs.append(pd.read_csv(partial))

    if not dfs:
        print("No results produced — check logs for errors.")
        return

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["image_id", "lambda"]).reset_index(drop=True)

    csv_path = args.output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows → {csv_path}")

    for metric in _PLOT_METRICS:
        _plot_gap(df, metric, args.output_dir)

    _write_summary(df, args.output_dir)

    # ---- Upload final aggregate outputs to R2 ------------------------------
    r2 = R2Client()
    aggregate_files = [args.output_dir / "results.csv", args.output_dir / "SUMMARY.md"]
    for metric in _PLOT_METRICS:
        for ext in ("png", "pdf"):
            aggregate_files.append(args.output_dir / f"gap_{metric}.{ext}")

    for fpath in aggregate_files:
        if fpath.exists():
            r2.upload_file(fpath, f"{r2_prefix}/{fpath.name}")

    print(f"\nAll outputs uploaded to R2: {r2_prefix}/")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
