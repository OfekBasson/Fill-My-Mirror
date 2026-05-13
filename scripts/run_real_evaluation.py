"""
Real-images dataset evaluation script — compares multiple models across multiple seeds.

All R2 paths share the prefix: real/estimated_geometry/

Models evaluated:
  Ours (with interpolation):
    real/estimated_geometry/black-forest-labs--FLUX.1-Fill-dev/n_13.0_t_625.0/<idx>/seed_<s>.png
  Ours (no interpolation):
    real/estimated_geometry/black-forest-labs--FLUX.1-Fill-dev/n_13.0_t_0.0/<idx>/seed_<s>.png
  FLUX.1 Fill (vanilla):
    real/estimated_geometry/black-forest-labs--FLUX.1-Fill-dev_vanilla/<idx>/seed_<s>.png
  FLUX.2 Klein (vanilla):
    real/estimated_geometry/black-forest-labs--FLUX.2-klein-base-9B_vanilla/<idx>/seed_<s>.png
  Qwen-2511 (vanilla):
    real/estimated_geometry/Qwen--Qwen-Image-Edit-2511/<idx>/seed_<s>.png
  MirrorFusion:
    real/estimated_geometry/mirrorfusion_depth_concat/<idx>/seed_<s>.png

Seeds evaluated: 0, 42, 512

Reference masks per index:
  Full mirror mask : real/estimated_geometry/<idx>/generative_refinement_mask.png
  Constrained mask : real/estimated_geometry/<idx>/rcs_mask.png
    (computed on-the-fly via MASt3R RCS if not already in R2)

Ground-truth images:
  real/estimated_geometry/<idx>/gt_image.png

Outputs (written to --output-dir and uploaded to R2 under real/evaluation/):
  missing_indices.txt
  <model_slug>/
    <index>_seed_<seed>_metrics.json
    summary.json
  comparison_table.csv
  comparison_plots.pdf
  seed_variance_plots.pdf

Usage
-----
Evaluate all models (skipping per-index JSONs that already exist):
    python scripts/run_real_evaluation.py --skip-existing

Evaluate specific seeds only:
    python scripts/run_real_evaluation.py --seeds 0 42

Regenerate summaries and plots from existing per-index JSONs:
    python scripts/run_real_evaluation.py --plots-only

Specify a custom output directory:
    python scripts/run_real_evaluation.py --output-dir outputs/real_eval/
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from fill_my_mirror.evaluation.metrics_computation import GeneratedImage, MetricsInput, compute_metrics
from fill_my_mirror.storage import R2Client

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R2_BASE_PREFIX = "real/estimated_geometry"
R2_EVAL_PREFIX = "real/evaluation"
RCS_COMPUTE_SIZE = 800  # all RCS computation (dilation included) runs at this resolution

SEEDS: list[int] = [0, 42, 512]

METRIC_KEYS = (
    "psnr_full_mirror", "ssim_full_mirror", "lpips_full_mirror",
    "psnr_constrained", "ssim_constrained", "lpips_constrained",
    "clip_similarity",
)

MODELS: list[dict] = [
    {
        "name": "Ours (interpolation)",
        "slug": "ours_interp",
        "key_template": f"{R2_BASE_PREFIX}/black-forest-labs--FLUX.1-Fill-dev/n_13.0_t_625.0/{{index}}/seed_{{seed}}.png",
    },
    {
        "name": "Ours (no interpolation)",
        "slug": "ours_no_interp",
        "key_template": f"{R2_BASE_PREFIX}/black-forest-labs--FLUX.1-Fill-dev/n_13.0_t_0.0/{{index}}/seed_{{seed}}.png",
    },
    {
        "name": "FLUX.1 Fill",
        "slug": "flux1_fill_vanilla",
        "key_template": f"{R2_BASE_PREFIX}/black-forest-labs--FLUX.1-Fill-dev_vanilla/{{index}}/seed_{{seed}}.png",
    },
    {
        "name": "FLUX.2 Klein",
        "slug": "flux2_klein_vanilla",
        "key_template": f"{R2_BASE_PREFIX}/black-forest-labs--FLUX.2-klein-base-9B_vanilla/{{index}}/seed_{{seed}}.png",
    },
    {
        "name": "Qwen-2511",
        "slug": "qwen_2511_vanilla",
        "key_template": f"{R2_BASE_PREFIX}/Qwen--Qwen-Image-Edit-2511/{{index}}/seed_{{seed}}.png",
    },
    {
        "name": "MirrorFusion",
        "slug": "mirrorfusion",
        "key_template": f"{R2_BASE_PREFIX}/mirrorfusion_depth_concat/{{index}}/seed_{{seed}}.png",
    },
]

# ---------------------------------------------------------------------------
# RCS computation (for constrained mask)
# ---------------------------------------------------------------------------

def _compute_rcs_mask(
    image_pil: Image.Image,
    mirror_mask_pil: Image.Image,
    mast3r_model_name: str,
    device: str,
    dilation_radius: int = 4,
    dilation_iterations: int = 1,
) -> np.ndarray:
    """Compute RCS mask (union of hflip + rot180 correspondences, dilated & intersected)."""
    from fill_my_mirror.evaluation.rcs_mask_computation import (
        _run_mast3r_correspondences,
        _dilate_and_intersect,
        _pil_to_uint8_rgb,
        _pil_to_binary,
        _MAST3R_IMG_SIZE,
    )

    image_arr = _pil_to_uint8_rgb(image_pil)
    mirror_mask_orig = _pil_to_binary(mirror_mask_pil)
    H, W = image_arr.shape[:2]
    S = _MAST3R_IMG_SIZE

    CS = RCS_COMPUTE_SIZE
    image_cs = np.array(Image.fromarray(image_arr).resize((CS, CS), Image.LANCZOS))
    mirror_mask = np.array(
        Image.fromarray(mirror_mask_orig.astype(np.uint8) * 255).resize((CS, CS), Image.NEAREST)
    ) > 127

    scene = image_cs.copy()
    scene[mirror_mask] = 0

    combined_cs = np.zeros((CS, CS), dtype=bool)
    for transform in ("hflip", "rot180"):
        mirror = image_cs.copy()
        mirror[~mirror_mask] = 0
        if transform == "hflip":
            mirror_t = np.fliplr(mirror)
        else:
            mirror_t = np.rot90(mirror, 2)

        pts_scene, pts_mirror = _run_mast3r_correspondences(
            scene, mirror_t, mast3r_model_name, device,
        )
        if pts_mirror.shape[0] == 0:
            continue

        x, y = pts_mirror[:, 0].copy(), pts_mirror[:, 1].copy()
        if transform == "hflip":
            x = S - 1 - x
        elif transform == "rot180":
            x = S - 1 - x
            y = S - 1 - y

        xs = np.clip(np.round(x * (CS / S)).astype(int), 0, CS - 1)
        ys = np.clip(np.round(y * (CS / S)).astype(int), 0, CS - 1)
        combined_cs[ys, xs] = True

    rcs_cs = _dilate_and_intersect(combined_cs, mirror_mask, dilation_radius, iterations=dilation_iterations)

    return np.array(
        Image.fromarray(rcs_cs.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
    ) > 127


def _ensure_rcs_mask(
    index: int,
    r2: R2Client,
    tmp_root: Path,
    mast3r_model_name: str,
    device: str,
) -> Image.Image | None:
    """
    Return the RCS mask PIL image for this index.
    Downloads from R2 if already computed; otherwise computes it, saves locally,
    and uploads to R2 under real/estimated_geometry/<index>/rcs_mask.png.
    """
    rcs_r2_key = f"{R2_BASE_PREFIX}/{index}/rcs_mask.png"

    if r2.key_exists(rcs_r2_key):
        d = tmp_root / str(index)
        d.mkdir(parents=True, exist_ok=True)
        rcs_local = d / "rcs_mask.png"
        try:
            r2.download_file(rcs_r2_key, rcs_local)
            logger.info("[%d] Loaded RCS mask from R2.", index)
            return Image.open(rcs_local).copy()
        except Exception as e:
            logger.warning("[%d] Failed to download existing RCS mask: %s — recomputing.", index, e)

    from fill_my_mirror.evaluation.rcs_mask_computation import _ensure_mast3r
    if not _ensure_mast3r():
        logger.error("[%d] MASt3R not available — cannot compute RCS mask.", index)
        return None

    d = tmp_root / str(index)
    d.mkdir(parents=True, exist_ok=True)
    img_local = d / "gt_image.png"
    mirror_local = d / "generative_refinement_mask.png"
    try:
        r2.download_file(f"{R2_BASE_PREFIX}/{index}/gt_image.png", img_local)
        r2.download_file(f"{R2_BASE_PREFIX}/{index}/generative_refinement_mask.png", mirror_local)
    except Exception as e:
        logger.warning("[%d] Cannot download inputs for RCS computation: %s", index, e)
        return None

    image_pil = Image.open(img_local).copy()
    mirror_pil = Image.open(mirror_local).copy()

    try:
        rcs_arr = _compute_rcs_mask(image_pil, mirror_pil, mast3r_model_name, device)
    except Exception:
        logger.warning("[%d] RCS computation failed.", index)
        traceback.print_exc()
        return None

    rcs_pil = Image.fromarray((rcs_arr.astype(np.uint8) * 255), mode="L")
    rcs_local = d / "rcs_mask.png"
    rcs_pil.save(rcs_local)

    try:
        r2.upload_file(rcs_local, rcs_r2_key)
        logger.info("[%d] Uploaded RCS mask to %s", index, rcs_r2_key)
    except Exception as e:
        logger.warning("[%d] Failed to upload RCS mask: %s", index, e)

    return rcs_pil


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r2_metrics_key(model_slug: str, index: int, seed: int) -> str:
    return f"{R2_EVAL_PREFIX}/{model_slug}/{index}_seed_{seed}_metrics.json"


def _upload_json(data: dict | list, r2_key: str, r2: R2Client) -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        json.dump(data, tf, indent=2)
        tf_path = Path(tf.name)
    try:
        r2.upload_file(tf_path, r2_key)
    finally:
        tf_path.unlink(missing_ok=True)


def _save_json_local(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Step 1 — Discover which (index, seed) pairs exist per model
# ---------------------------------------------------------------------------

def discover_indices(r2: R2Client, seeds: list[int]) -> dict[str, dict[int, set[int]]]:
    """Returns {slug: {index: {seeds_present}}}."""
    present: dict[str, dict[int, set[int]]] = {m["slug"]: defaultdict(set) for m in MODELS}

    prefixes_to_models: dict[str, list[str]] = defaultdict(list)
    for m in MODELS:
        parent = m["key_template"].split("/{index}/")[0]
        prefixes_to_models[parent].append(m["slug"])

    seed_filenames = {f"seed_{s}.png" for s in seeds}

    for prefix, slugs in prefixes_to_models.items():
        print(f"  Listing {prefix}/...")
        keys = r2.list_keys(prefix + "/")
        prefix_depth = len(prefix.split("/"))
        for key in keys:
            parts = key.split("/")
            if len(parts) <= prefix_depth:
                continue
            idx_str = parts[prefix_depth]
            if not idx_str.isdigit():
                continue
            filename = parts[-1]
            if filename not in seed_filenames:
                continue
            seed_val = int(filename.replace("seed_", "").replace(".png", ""))
            idx = int(idx_str)
            for slug in slugs:
                model = next(m for m in MODELS if m["slug"] == slug)
                expected = model["key_template"].format(index=idx, seed=seed_val)
                if key == expected:
                    present[slug][idx].add(seed_val)

    # Convert defaultdict to regular dict
    return {slug: dict(d) for slug, d in present.items()}


# ---------------------------------------------------------------------------
# Step 2 — Missing-indices report
# ---------------------------------------------------------------------------

def write_missing_report(
    present: dict[str, dict[int, set[int]]],
    seeds: list[int],
    output_dir: Path,
) -> None:
    all_indices = sorted({idx for d in present.values() for idx in d})
    total = len(all_indices)
    lines: list[str] = [
        f"Real dataset — missing indices report (union of all models: {total} images, seeds: {seeds})\n"
    ]

    for m in MODELS:
        slug = m["slug"]
        model_present = present[slug]
        missing_any: list[int] = []
        missing_seeds: dict[int, list[int]] = {}
        for idx in all_indices:
            if idx not in model_present:
                missing_any.append(idx)
            else:
                ms = sorted(set(seeds) - model_present[idx])
                if ms:
                    missing_seeds[idx] = ms
        lines.append(f"\n{'=' * 70}")
        lines.append(f"Model : {m['name']}  ({slug})")
        lines.append(f"Found at least one seed: {len(model_present)} / {total}")
        lines.append(f"Missing entirely ({len(missing_any)}): "
                     f"{missing_any if len(missing_any) <= 50 else str(missing_any[:50]) + ' ...'}")
        if missing_seeds:
            sample = list(missing_seeds.items())[:10]
            lines.append(f"Partially missing seeds (first 10): {sample}")

    report = "\n".join(lines) + "\n"
    local_path = output_dir / "missing_indices.txt"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(report)
    print(f"  Saved {local_path}")


# ---------------------------------------------------------------------------
# Step 3 — Mask cache (full mirror + RCS constrained mask)
# ---------------------------------------------------------------------------

class _MaskCache:
    def __init__(self, r2: R2Client, tmp_root: Path, mast3r_model_name: str, device: str):
        self._r2 = r2
        self._tmp_root = tmp_root
        self._mast3r_model_name = mast3r_model_name
        self._device = device
        self._cache: dict[int, dict | None] = {}

    def get(self, index: int) -> dict | None:
        if index in self._cache:
            return self._cache[index]

        d = self._tmp_root / str(index)
        d.mkdir(parents=True, exist_ok=True)

        full_mirror_key = f"{R2_BASE_PREFIX}/{index}/generative_refinement_mask.png"
        full_local = d / "generative_refinement_mask.png"
        try:
            self._r2.download_file(full_mirror_key, full_local)
            full_mirror_mask = Image.open(full_local).copy()
        except Exception:
            logger.warning("[%d] Cannot download full mirror mask.", index)
            self._cache[index] = None
            return None

        rcs_pil = _ensure_rcs_mask(
            index, self._r2, self._tmp_root / "rcs",
            self._mast3r_model_name, self._device,
        )
        if rcs_pil is None:
            logger.warning("[%d] RCS mask unavailable — skipping.", index)
            self._cache[index] = None
            return None

        result = {"full_mirror_mask": full_mirror_mask, "constrained_mask": rcs_pil}
        self._cache[index] = result
        return result


# ---------------------------------------------------------------------------
# Step 4 — Per-(index, seed) metric computation
# ---------------------------------------------------------------------------

def evaluate_index_all_models(
    active_models: list[dict],
    index: int,
    seeds: list[int],
    gt_image: Image.Image,
    masks: dict,
    r2: R2Client,
    tmp_root: Path,
    skip_existing: bool,
    output_dir: Path,
) -> dict[str, dict[int, dict]]:
    """Returns {slug: {seed: metrics_dict}}."""
    results: dict[str, dict[int, dict]] = {m["slug"]: {} for m in active_models}

    for seed in seeds:
        pending: list[dict] = []
        cached_seed: dict[str, dict] = {}

        for m in active_models:
            r2_key = _r2_metrics_key(m["slug"], index, seed)
            if skip_existing and r2.key_exists(r2_key):
                try:
                    with tempfile.NamedTemporaryFile(suffix=".json") as tf:
                        r2.download_file(r2_key, Path(tf.name))
                        cached_seed[m["slug"]] = json.loads(Path(tf.name).read_text())
                    continue
                except Exception:
                    pass
            pending.append(m)

        for slug, metrics in cached_seed.items():
            results[slug][seed] = metrics

        if not pending:
            continue

        img_dir = tmp_root / str(index) / f"seed_{seed}"
        img_dir.mkdir(parents=True, exist_ok=True)

        gen_images: list[GeneratedImage] = []
        slug_to_local: dict[str, Path] = {}

        for m in pending:
            img_key = m["key_template"].format(index=index, seed=seed)
            local_img = img_dir / f"{m['slug']}.png"
            try:
                r2.download_file(img_key, local_img)
                gen_images.append(GeneratedImage(
                    name=m["slug"],
                    image=Image.open(local_img).convert("RGB"),
                ))
                slug_to_local[m["slug"]] = local_img
            except Exception:
                print(f"    [eval] cannot download {img_key}")

        if not gen_images:
            continue

        try:
            with tempfile.TemporaryDirectory() as tmp_metrics:
                df = compute_metrics(MetricsInput(
                    gt_image=gt_image,
                    generated_images=gen_images,
                    full_mirror_mask=masks["full_mirror_mask"],
                    constrained_mask=masks["constrained_mask"],
                    save_path=tmp_metrics,
                    prompt="",
                ))
        except Exception:
            print(f"    [eval] compute_metrics failed for index {index} seed {seed}")
            traceback.print_exc()
            for p in slug_to_local.values():
                p.unlink(missing_ok=True)
            continue

        for _, row in df.iterrows():
            slug = row["name"]
            metrics = row.drop("name").to_dict()
            _save_json_local(metrics, output_dir / slug / f"{index}_seed_{seed}_metrics.json")
            _upload_json(metrics, _r2_metrics_key(slug, index, seed), r2)
            results[slug][seed] = metrics

        for p in slug_to_local.values():
            p.unlink(missing_ok=True)

    return results


# ---------------------------------------------------------------------------
# Step 5 — Aggregate statistics (multi-seed aware)
# ---------------------------------------------------------------------------

def _agg_stats(values: list[float]) -> dict:
    a = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(a) == 0:
        return {"mean": None, "median": None, "std": None, "se": None, "count": 0}
    return {
        "mean":   float(np.mean(a)),
        "median": float(np.median(a)),
        "std":    float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "se":     float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
        "count":  int(len(a)),
    }


def build_summary(
    model: dict,
    index_seed_metrics: dict[int, dict[int, dict]],
    seeds: list[int],
) -> dict:
    """
    Aggregates across all (index, seed) pairs.

    Strategy:
    - For each index, compute the mean across available seeds → one value per index.
    - Then aggregate these per-index means across all indices.
    - Additionally report seed-level variance: for each index that has all seeds,
      compute std across seeds; then report mean of those stds.
    """
    # Per-metric collections
    per_metric_means: dict[str, list[float]] = {m: [] for m in METRIC_KEYS}

    # Seed-variance: per-metric, list of within-index stds (only where ≥2 seeds)
    per_metric_seed_stds: dict[str, list[float]] = {m: [] for m in METRIC_KEYS}

    # Per-seed aggregation (for seed-level breakdown table)
    per_seed_values: dict[int, dict[str, list[float]]] = {
        s: {m: [] for m in METRIC_KEYS} for s in seeds
    }

    n_indices_with_all_seeds = 0

    for index, seed_metrics in index_seed_metrics.items():
        for metric in METRIC_KEYS:
            seed_vals = [
                seed_metrics[s][metric]
                for s in seeds
                if s in seed_metrics and seed_metrics[s].get(metric) is not None
            ]
            if not seed_vals:
                continue
            per_metric_means[metric].append(float(np.mean(seed_vals)))
            if len(seed_vals) >= 2:
                per_metric_seed_stds[metric].append(float(np.std(seed_vals, ddof=1)))

        if len(seed_metrics) == len(seeds):
            n_indices_with_all_seeds += 1

        for s, metrics in seed_metrics.items():
            if s not in per_seed_values:
                continue
            for metric in METRIC_KEYS:
                v = metrics.get(metric)
                if v is not None:
                    per_seed_values[s][metric].append(v)

    return {
        "model_name": model["name"],
        "model_slug": model["slug"],
        "num_indices_evaluated": len(index_seed_metrics),
        "num_indices_with_all_seeds": n_indices_with_all_seeds,
        "seeds_evaluated": seeds,
        # Main result: mean-over-seeds per index, then aggregated across indices
        "aggregated": {m: _agg_stats(vals) for m, vals in per_metric_means.items()},
        # Seed variance: mean within-index std across seeds
        "seed_variance": {
            m: {
                "mean_within_index_std": float(np.mean(stds)) if stds else None,
                "count": len(stds),
            }
            for m, stds in per_metric_seed_stds.items()
        },
        # Per-seed breakdown
        "per_seed": {
            str(s): {m: _agg_stats(vals) for m, vals in per_seed_values[s].items()}
            for s in seeds
        },
    }


# ---------------------------------------------------------------------------
# Step 6 — Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(summaries: list[dict]) -> pd.DataFrame:
    rows = []
    for s in summaries:
        row = {
            "model": s["model_name"],
            "n_indices": s["num_indices_evaluated"],
            "n_all_seeds": s["num_indices_with_all_seeds"],
        }
        for m in METRIC_KEYS:
            agg = s["aggregated"].get(m, {})
            row[f"{m}_mean"] = agg.get("mean")
            row[f"{m}_se"]   = agg.get("se")
            sv = s["seed_variance"].get(m, {})
            row[f"{m}_seed_std"] = sv.get("mean_within_index_std")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 7 — Plots
# ---------------------------------------------------------------------------

_PLOT_METRICS = [
    ("psnr_full_mirror",   "PSNR full mirror (dB)"),
    ("psnr_constrained",   "PSNR constrained (dB)"),
    ("ssim_full_mirror",   "SSIM full mirror"),
    ("ssim_constrained",   "SSIM constrained"),
    ("lpips_full_mirror",  "LPIPS full mirror ↓"),
    ("lpips_constrained",  "LPIPS constrained ↓"),
]


_PLOT_EXCLUDE_SLUGS = {"flux2_klein_vanilla"}


def build_plots(summaries: list[dict], output_dir: Path, r2: R2Client) -> None:
    """Bar chart: mean±SE across all (index, seed) pairs."""
    summaries = [s for s in summaries if s["model_slug"] not in _PLOT_EXCLUDE_SLUGS]
    model_names = [s["model_name"] for s in summaries]
    xs = np.arange(len(model_names))
    width = 0.6

    n_metrics = len(_PLOT_METRICS)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(max(10, len(model_names) * 1.4 + 2), 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(model_names))]

    for ax, (metric, ylabel) in zip(axes, _PLOT_METRICS):
        means = np.array([s["aggregated"].get(metric, {}).get("mean") or float("nan") for s in summaries])
        ses   = np.array([s["aggregated"].get(metric, {}).get("se")   or 0.0           for s in summaries])

        ax.bar(xs, np.where(np.isnan(means), 0, means), width, color=colors, alpha=0.85, zorder=3)
        ax.errorbar(xs, np.where(np.isnan(means), 0, means), yerr=ses,
                    fmt="none", color="black", capsize=4, capthick=1.2, zorder=4)

        for i, m in enumerate(means):
            if np.isnan(m):
                ax.text(xs[i], 0.01, "N/A", ha="center", va="bottom", fontsize=7, color="gray",
                        transform=ax.get_xaxis_transform())

        ax.set_xticks(xs)
        ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.set_xlim(-0.5, len(model_names) - 0.5)

    fig.tight_layout()

    local_pdf = output_dir / "comparison_plots.pdf"
    fig.savefig(local_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {local_pdf}")
    r2.upload_file(local_pdf, f"{R2_EVAL_PREFIX}/comparison_plots.pdf")
    print(f"  Uploaded {R2_EVAL_PREFIX}/comparison_plots.pdf")


def build_seed_variance_plots(summaries: list[dict], output_dir: Path, r2: R2Client) -> None:
    """Two-panel plots: per-seed breakdown + within-index seed variance."""
    summaries = [s for s in summaries if s["model_slug"] not in _PLOT_EXCLUDE_SLUGS]
    seeds_str = summaries[0].get("seeds_evaluated", []) if summaries else []
    if not seeds_str:
        return

    model_names = [s["model_name"] for s in summaries]
    n_models = len(model_names)
    cmap = plt.get_cmap("Set2")

    for metric, ylabel in _PLOT_METRICS:
        fig, (ax_seeds, ax_var) = plt.subplots(1, 2, figsize=(max(14, n_models * 2), 5))

        # Left: per-seed means grouped by model
        n_seeds = len(seeds_str)
        group_width = 0.7
        bar_w = group_width / n_seeds
        xs = np.arange(n_models)

        for si, seed in enumerate(seeds_str):
            seed_key = str(seed)
            vals = np.array([
                s["per_seed"].get(seed_key, {}).get(metric, {}).get("mean") or float("nan")
                for s in summaries
            ])
            offset = (si - (n_seeds - 1) / 2) * bar_w
            ax_seeds.bar(xs + offset, np.where(np.isnan(vals), 0, vals),
                         bar_w * 0.9, label=f"seed {seed}",
                         color=cmap(si % 8), alpha=0.85, zorder=3)

        ax_seeds.set_xticks(xs)
        ax_seeds.set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)
        ax_seeds.set_ylabel(ylabel, fontsize=9)
        ax_seeds.set_title("Per-seed breakdown", fontsize=10)
        ax_seeds.legend(fontsize=8)
        ax_seeds.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

        # Right: mean within-index std across seeds (lower = more stable)
        stds = np.array([
            s["seed_variance"].get(metric, {}).get("mean_within_index_std") or float("nan")
            for s in summaries
        ])
        bar_colors = [cmap(i % 8) for i in range(n_models)]
        ax_var.bar(xs, np.where(np.isnan(stds), 0, stds), 0.6, color=bar_colors, alpha=0.85, zorder=3)
        for i, v in enumerate(stds):
            if np.isnan(v):
                ax_var.text(i, 0.0, "N/A", ha="center", va="bottom", fontsize=7, color="gray",
                            transform=ax_var.get_xaxis_transform())
        ax_var.set_xticks(xs)
        ax_var.set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)
        ax_var.set_ylabel(f"Mean within-index σ ({ylabel.split('(')[0].strip()})", fontsize=9)
        ax_var.set_title("Seed variance (lower = more stable)", fontsize=10)
        ax_var.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

        fig.tight_layout()

        safe_metric = metric.replace("/", "_")
        local_pdf = output_dir / f"seed_variance_{safe_metric}.pdf"
        fig.savefig(local_pdf, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {local_pdf}")

        # Save each panel separately (no title, no axis titles)
        for panel_ax, panel_suffix in [(ax_seeds, "per_seed"), (ax_var, "seed_std")]:
            fig_single, ax_single = plt.subplots(1, 1, figsize=(max(7, n_models * 1.0), 5))
            for artist in panel_ax.get_children():
                pass  # copy via re-drawing below
            # Re-draw the panel independently
            fig_single2, ax_single2 = plt.subplots(1, 1, figsize=(max(7, n_models * 1.0), 5))
            plt.close(fig_single)
            if panel_suffix == "per_seed":
                for si, seed in enumerate(seeds_str):
                    seed_key = str(seed)
                    vals = np.array([
                        s["per_seed"].get(seed_key, {}).get(metric, {}).get("mean") or float("nan")
                        for s in summaries
                    ])
                    offset = (si - (n_seeds - 1) / 2) * bar_w
                    ax_single2.bar(xs + offset, np.where(np.isnan(vals), 0, vals),
                                   bar_w * 0.9, label=f"seed {seed}",
                                   color=cmap(si % 8), alpha=0.85, zorder=3)
                ax_single2.set_xticks(xs)
                ax_single2.set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)
                ax_single2.set_ylabel(ylabel, fontsize=9)
                ax_single2.legend(fontsize=8)
                ax_single2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
            else:
                ax_single2.bar(xs, np.where(np.isnan(stds), 0, stds), 0.6, color=bar_colors, alpha=0.85, zorder=3)
                for i, v in enumerate(stds):
                    if np.isnan(v):
                        ax_single2.text(i, 0.0, "N/A", ha="center", va="bottom", fontsize=7, color="gray",
                                        transform=ax_single2.get_xaxis_transform())
                ax_single2.set_xticks(xs)
                ax_single2.set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)
                ax_single2.set_ylabel(f"Mean within-index σ ({ylabel.split('(')[0].strip()})", fontsize=9)
                ax_single2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
            fig_single2.tight_layout()
            panel_pdf = output_dir / f"seed_variance_{safe_metric}_{panel_suffix}.pdf"
            fig_single2.savefig(panel_pdf, format="pdf", bbox_inches="tight")
            plt.close(fig_single2)
            print(f"  Saved {panel_pdf}")

    # Also build a combined seed-variance summary PDF
    fig, axes = plt.subplots(len(_PLOT_METRICS), 1,
                             figsize=(max(10, n_models * 1.4 + 2), 4 * len(_PLOT_METRICS)))
    if len(_PLOT_METRICS) == 1:
        axes = [axes]

    for ax, (metric, ylabel) in zip(axes, _PLOT_METRICS):
        stds = np.array([
            s["seed_variance"].get(metric, {}).get("mean_within_index_std") or float("nan")
            for s in summaries
        ])
        xs = np.arange(n_models)
        bar_colors = [plt.get_cmap("tab10")(i % 10) for i in range(n_models)]
        ax.bar(xs, np.where(np.isnan(stds), 0, stds), 0.6, color=bar_colors, alpha=0.85, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(f"Mean σ across seeds\n({ylabel})", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    fig.tight_layout()
    local_pdf = output_dir / "seed_variance_plots.pdf"
    fig.savefig(local_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {local_pdf}")
    r2.upload_file(local_pdf, f"{R2_EVAL_PREFIX}/seed_variance_plots.pdf")
    print(f"  Uploaded {R2_EVAL_PREFIX}/seed_variance_plots.pdf")


# ---------------------------------------------------------------------------
# Helpers for plots-only / partial-run paths
# ---------------------------------------------------------------------------

def _check_all_evaluated(
    active_models: list[dict],
    present: dict[str, dict[int, set[int]]],
    seeds: list[int],
    r2: R2Client,
) -> bool:
    all_done = True
    for m in active_models:
        missing = [
            (idx, s)
            for idx, s_set in present[m["slug"]].items()
            for s in seeds
            if s in s_set and not r2.key_exists(_r2_metrics_key(m["slug"], idx, s))
        ]
        if missing:
            print(f"  [{m['name']}] {len(missing)} (index,seed) pairs not yet evaluated "
                  f"(e.g. {missing[:5]})")
            all_done = False
    return all_done


def _load_existing_metrics(
    model: dict,
    present: dict[int, set[int]],
    seeds: list[int],
    r2: R2Client,
) -> dict[int, dict[int, dict]]:
    result: dict[int, dict[int, dict]] = {}
    for index in sorted(present):
        seed_metrics: dict[int, dict] = {}
        for seed in seeds:
            if seed not in present[index]:
                continue
            r2_key = _r2_metrics_key(model["slug"], index, seed)
            try:
                with tempfile.NamedTemporaryFile(suffix=".json") as tf:
                    r2.download_file(r2_key, Path(tf.name))
                    seed_metrics[seed] = json.loads(Path(tf.name).read_text())
            except Exception:
                pass
        if seed_metrics:
            result[index] = seed_metrics
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple models on the real-images dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_eval"),
                        help="Local directory for outputs (default: outputs/real_eval/)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip (index, seed) pairs whose per-index JSON already exists in R2")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip metric computation; reload existing per-index JSONs from R2")
    parser.add_argument("--models", nargs="+", metavar="SLUG",
                        help="Restrict evaluation to specific model slugs (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help=f"Seeds to evaluate (default: {SEEDS})")
    parser.add_argument("--start-index", type=int, default=None,
                        help="First index to evaluate (inclusive).")
    parser.add_argument("--end-index", type=int, default=None,
                        help="Last index to evaluate (exclusive).")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device for MASt3R (cuda/cpu). Default: auto-detect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds: list[int] = sorted(set(args.seeds))
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    active_models = MODELS
    if args.models:
        active_models = [m for m in MODELS if m["slug"] in args.models]
        if not active_models:
            print(f"No models matched slugs: {args.models}")
            return

    r2 = R2Client()

    print(f"Discovering available indices in R2 (seeds: {seeds})...")
    present = discover_indices(r2, seeds)

    print("\nWriting missing-indices report...")
    write_missing_report(present, seeds, output_dir)

    all_summaries: list[dict] = []

    if args.plots_only:
        print("\n--plots-only: loading existing per-index metrics from R2...")
        for m in active_models:
            slug = m["slug"]
            n = len(present[slug])
            print(f"\n  Loading {m['name']} ({n} indices)...")
            index_seed_metrics = _load_existing_metrics(m, present[slug], seeds, r2)
            summary = build_summary(m, index_seed_metrics, seeds)
            _save_json_local(summary, output_dir / slug / "summary.json")
            _upload_json(summary, f"{R2_EVAL_PREFIX}/{slug}/summary.json", r2)
            all_summaries.append(summary)
    else:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)
        mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

        all_active_indices = sorted({idx for m in active_models for idx in present[m["slug"]]})
        if args.start_index is not None or args.end_index is not None:
            all_active_indices = [
                idx for idx in all_active_indices
                if (args.start_index is None or idx >= args.start_index)
                and (args.end_index   is None or idx <  args.end_index)
            ]
            print(f"  Sliced to indices [{args.start_index}, {args.end_index}): {len(all_active_indices)} indices")
        else:
            print(f"  {len(all_active_indices)} unique indices across active models")

        # {slug: {index: {seed: metrics}}}
        index_seed_metrics_by_model: dict[str, dict[int, dict[int, dict]]] = {
            m["slug"]: {} for m in active_models
        }

        with tempfile.TemporaryDirectory(prefix="real_eval_") as tmp_str:
            tmp_root = Path(tmp_str)
            mask_cache = _MaskCache(r2, tmp_root / "masks", mast3r_model_name, device)

            for i, index in enumerate(all_active_indices):
                masks = mask_cache.get(index)
                if masks is None:
                    print(f"  [{i+1}/{len(all_active_indices)}] index {index}: masks unavailable — skipping")
                    continue

                gt_key = f"{R2_BASE_PREFIX}/{index}/gt_image.png"
                gt_local = tmp_root / "gt" / f"{index}.png"
                gt_local.parent.mkdir(parents=True, exist_ok=True)
                try:
                    r2.download_file(gt_key, gt_local)
                    gt_image = Image.open(gt_local).convert("RGB")
                except Exception:
                    print(f"  [{i+1}/{len(all_active_indices)}] index {index}: failed to load GT — skipping")
                    traceback.print_exc()
                    continue

                # Only include models that have at least one seed for this index
                models_for_index = [
                    m for m in active_models
                    if index in present[m["slug"]] and present[m["slug"]][index]
                ]
                # Filter seeds to those present for at least one model at this index
                seeds_for_index = [
                    s for s in seeds
                    if any(s in present[m["slug"]].get(index, set()) for m in models_for_index)
                ]

                results = evaluate_index_all_models(
                    models_for_index, index, seeds_for_index, gt_image, masks,
                    r2=r2,
                    tmp_root=tmp_root / "imgs",
                    skip_existing=args.skip_existing,
                    output_dir=output_dir,
                )
                for slug, seed_metrics in results.items():
                    if seed_metrics:
                        index_seed_metrics_by_model[slug][index] = seed_metrics

                gt_local.unlink(missing_ok=True)

                if (i + 1) % 50 == 0 or (i + 1) == len(all_active_indices):
                    counts = ", ".join(
                        f"{m['slug']}:{len(index_seed_metrics_by_model[m['slug']])}"
                        for m in active_models
                    )
                    print(f"  [{i+1}/{len(all_active_indices)}] evaluated — {counts}")

        is_partial_run = args.start_index is not None or args.end_index is not None
        if is_partial_run:
            print("\nPartial run — checking whether all indices are evaluated in R2 before summarizing...")
            if not _check_all_evaluated(active_models, present, seeds, r2):
                print("Not all indices evaluated yet — skipping summaries and plots.")
                print("Run without --start-index/--end-index (or with --plots-only) once all shards are done.")
                return

        for m in active_models:
            summary = build_summary(m, index_seed_metrics_by_model[m["slug"]], seeds)
            _save_json_local(summary, output_dir / m["slug"] / "summary.json")
            _upload_json(summary, f"{R2_EVAL_PREFIX}/{m['slug']}/summary.json", r2)
            all_summaries.append(summary)
            print(f"  Summary saved for {m['name']}")

    if not all_summaries:
        print("No summaries built — nothing to plot")
        return

    print("\nBuilding comparison table...")
    table = build_comparison_table(all_summaries)
    table_path = output_dir / "comparison_table.csv"
    table.to_csv(table_path, index=False, float_format="%.4f")
    print(f"  Saved {table_path}")
    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.4f}" if x is not None else "N/A"))
    r2.upload_file(table_path, f"{R2_EVAL_PREFIX}/comparison_table.csv")
    print(f"  Uploaded {R2_EVAL_PREFIX}/comparison_table.csv")

    print("\nBuilding comparison plots...")
    build_plots(all_summaries, output_dir, r2)

    print("\nBuilding seed variance plots...")
    build_seed_variance_plots(all_summaries, output_dir, r2)

    print("\nDone.")


if __name__ == "__main__":
    main()
