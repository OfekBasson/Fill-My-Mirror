"""
Evaluation script for Fill My Mirror inpainting experiments.

Scans R2 for seed images under:
    blender/gt_geometry/black-forest-labs--FLUX.1-Fill-dev/n_<n>_t_<t>/0/seed_<seed>.png

For each index found, downloads the reference images and masks:
    blender/gt_geometry/<index>/gt_image.png          (required)
    blender/gt_geometry/<index>/projected_image.png   (required)
    blender/gt_geometry/<index>/constrained_pixels_gt_geometry_mask.png
    blender/gt_geometry/<index>/generative_refinement_mask.png

Computes PSNR / SSIM / LPIPS / CLIP metrics for each seed image against both
references, saving per-seed JSON results and per-(n,t) summary JSON files to R2.
Also saves PDF summary plots and an ablation summary JSON to R2.

Aggregation per (n, t):
    1. Average all seeds within each scene index.
    2. Compute mean and standard error across scene indices.

Outputs written to R2:
    …/<model_slug>/n_<n>_t_<t>/0/seed_<seed>_metrics_gt_image.json
    …/<model_slug>/n_<n>_t_<t>/0/seed_<seed>_metrics_projected_image.json
    …/<model_slug>/n_<n>_t_<t>/n_<n>_t_<t>_summary.json
    …/<model_slug>/ablation_summary.json
    …/<model_slug>/psnr_vs_gt.pdf
    …/<model_slug>/psnr_vs_projected.pdf

Examples
--------
Evaluate all results:

    python scripts/run_evaluation.py

Skip seeds whose per-seed JSON already exists in R2:

    python scripts/run_evaluation.py --skip-existing

Regenerate summaries and plots from existing per-seed JSONs:

    python scripts/run_evaluation.py --plots-only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fill_my_mirror.evaluation.metrics_computation import GeneratedImage, MetricsInput, compute_metrics
from fill_my_mirror.storage import R2Client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_SLUG = "black-forest-labs--FLUX.1-Fill-dev"
BASE_PREFIX = "blender/gt_geometry"
MODEL_PREFIX = f"{BASE_PREFIX}/{MODEL_SLUG}"
SEED_RE = re.compile(r"^seed_(\d+)\.png$")
NT_RE = re.compile(r"^n_([\d.]+)_t_(.+)$")

METRIC_KEYS = (
    "psnr_full_mirror", "ssim_full_mirror", "lpips_full_mirror",
    "psnr_constrained", "ssim_constrained", "lpips_constrained",
    "clip_similarity",
)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _index_mean(per_seed_metrics: list[dict], metric: str) -> float | None:
    """Mean of a metric across seeds for one scene index. Returns None if no data."""
    vals = [v for v in (e.get(metric) for e in per_seed_metrics) if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else None


def _aggregate_across_indices(index_means: list[float | None]) -> dict:
    """
    Given per-index means, compute the cross-index mean and standard error.
    Returns {"mean": ..., "se": ..., "num_indices": ...} (values None if no data).
    """
    vals = [v for v in index_means if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "se": None, "num_indices": 0}
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "se": se, "num_indices": n}


def _nt_sort_key(nt: tuple[str, str]) -> tuple:
    n, t = nt
    try:
        n_float = float(n)
    except ValueError:
        n_float = float("inf")
    try:
        t_float = float(t)
    except ValueError:
        t_float = float("inf")
    return (n_float, t_float)


def _nt_label(n: str, t: str) -> str:
    return f"n={n}, t'={t}"


# ---------------------------------------------------------------------------
# R2 reference cache (per index)
# ---------------------------------------------------------------------------

class _RefCache:
    """Downloads and caches per-index reference images and masks from R2."""

    def __init__(self, r2: R2Client, tmp_root: Path):
        self._r2 = r2
        self._tmp_root = tmp_root
        self._cache: dict[int, dict | None] = {}

    def get(self, index: int) -> dict | None:
        if index in self._cache:
            return self._cache[index]

        d = self._tmp_root / str(index)
        d.mkdir(parents=True, exist_ok=True)
        prefix = f"{BASE_PREFIX}/{index}"
        refs: dict = {}
        ok = True

        for fname in (
            "gt_image.png",
            "projected_image.png",
            "constrained_pixels_gt_geometry_mask.png",
            "generative_refinement_mask.png",
        ):
            key = f"{prefix}/{fname}"
            local = d / fname
            try:
                self._r2.download_file(key, local)
                if fname.endswith("_image.png"):
                    refs[fname] = Image.open(local).convert("RGB")
                else:
                    refs[fname] = Image.open(local)
            except Exception:
                print(f"  [eval] index {index}: cannot download {fname} — skipping index")
                ok = False

        result = refs if ok else None
        self._cache[index] = result
        return result


# ---------------------------------------------------------------------------
# Core evaluation for one index
# ---------------------------------------------------------------------------

def _evaluate_index(
    index: int,
    seed_entries: list[dict],
    refs: dict,
    r2: R2Client,
    tmp_root: Path,
    skip_existing: bool,
) -> list[dict]:
    """
    Download seed images, compute metrics vs both references, upload per-seed JSONs.

    Returns a list of result dicts (one per seed entry) with keys:
        n, t, index, seed, metrics_gt, metrics_projected
    """
    results = []

    pending = []
    for entry in seed_entries:
        gt_key = entry["png_key"].replace(".png", "_metrics_gt_image.json")
        proj_key = entry["png_key"].replace(".png", "_metrics_projected_image.json")
        if skip_existing and r2.key_exists(gt_key) and r2.key_exists(proj_key):
            try:
                with tempfile.NamedTemporaryFile(suffix=".json") as tf:
                    r2.download_file(gt_key, Path(tf.name))
                    gt_data = json.loads(Path(tf.name).read_text())
                with tempfile.NamedTemporaryFile(suffix=".json") as tf:
                    r2.download_file(proj_key, Path(tf.name))
                    proj_data = json.loads(Path(tf.name).read_text())
                results.append({**entry, "metrics_gt": gt_data, "metrics_projected": proj_data})
                continue
            except Exception:
                pass
        pending.append(entry)

    if not pending:
        return results

    seed_dir = tmp_root / str(index)
    seed_dir.mkdir(parents=True, exist_ok=True)
    loaded: list[tuple[dict, Path]] = []
    for entry in pending:
        local = seed_dir / f"seed_{entry['seed']}_n{entry['n']}_t{entry['t']}.png"
        try:
            r2.download_file(entry["png_key"], local)
            loaded.append((entry, local))
        except Exception:
            print(f"  [eval] index {index}: cannot download {entry['png_key']} — skipping seed")

    if not loaded:
        return results

    gen_images = [
        GeneratedImage(
            name=f"n_{e['n']}_t_{e['t']}_seed_{e['seed']}",
            image=Image.open(p).convert("RGB"),
        )
        for e, p in loaded
    ]

    gt_pil = refs["gt_image.png"]
    proj_pil = refs["projected_image.png"]
    full_mirror_mask = refs["generative_refinement_mask.png"]
    constrained_mask = refs["constrained_pixels_gt_geometry_mask.png"]
    prompt = next((e["prompt"] for e, _ in loaded if e.get("prompt")), "")

    def _run_metrics(ref_pil: Image.Image) -> dict[str, dict]:
        with tempfile.TemporaryDirectory() as tmp:
            df = compute_metrics(MetricsInput(
                gt_image=ref_pil,
                generated_images=gen_images,
                full_mirror_mask=full_mirror_mask,
                constrained_mask=constrained_mask,
                save_path=tmp,
                prompt=prompt,
            ))
        return {row["name"]: row.drop("name").to_dict() for _, row in df.iterrows()}

    gt_results: dict[str, dict] = {}
    proj_results: dict[str, dict] = {}

    try:
        gt_results = _run_metrics(gt_pil)
    except Exception:
        print(f"  [eval] index {index}: metrics vs gt_image failed")
        traceback.print_exc()

    try:
        proj_results = _run_metrics(proj_pil)
    except Exception:
        print(f"  [eval] index {index}: metrics vs projected_image failed")
        traceback.print_exc()

    for entry, _ in loaded:
        name = f"n_{entry['n']}_t_{entry['t']}_seed_{entry['seed']}"
        gt_data = gt_results.get(name, {})
        proj_data = proj_results.get(name, {})

        for data, r2_key in [
            (gt_data,   entry["png_key"].replace(".png", "_metrics_gt_image.json")),
            (proj_data, entry["png_key"].replace(".png", "_metrics_projected_image.json")),
        ]:
            with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
                json.dump(data, tf, indent=2)
                tf_path = Path(tf.name)
            try:
                r2.upload_file(tf_path, r2_key)
            finally:
                tf_path.unlink(missing_ok=True)

        results.append({**entry, "metrics_gt": gt_data, "metrics_projected": proj_data})

    return results


# ---------------------------------------------------------------------------
# Per-(n,t) aggregated stats
# ---------------------------------------------------------------------------

def _compute_nt_stats(entries: list[dict]) -> dict:
    """
    Compute per-index seed-averaged metrics, then aggregate across indices.

    entries: all result dicts for one (n, t) combo.

    Returns a stats dict with structure:
        {
          "num_indices": int,
          "num_total_outputs": int,
          "seeds_per_index": {str(index): int, ...},
          "per_index": {str(index): {"metrics_gt": {...}, "metrics_projected": {...}}},
          "aggregated": {
            "gt":        {metric: {"mean": ..., "se": ..., "num_indices": ...}, ...},
            "projected": {metric: {"mean": ..., "se": ..., "num_indices": ...}, ...},
          }
        }
    """
    by_index: dict[int, list[dict]] = defaultdict(list)
    for e in entries:
        by_index[e["index"]].append(e)

    seeds_per_index = {str(idx): len(seeds) for idx, seeds in by_index.items()}

    per_index_avg: dict[str, dict] = {}
    for idx, seeds in by_index.items():
        per_index_avg[str(idx)] = {
            "metrics_gt":        {m: _index_mean([s["metrics_gt"]        for s in seeds], m) for m in METRIC_KEYS},
            "metrics_projected": {m: _index_mean([s["metrics_projected"] for s in seeds], m) for m in METRIC_KEYS},
        }

    aggregated: dict[str, dict] = {}
    for ref in ("gt", "projected"):
        ref_key = f"metrics_{ref}"
        aggregated[ref] = {
            m: _aggregate_across_indices([per_index_avg[str(idx)][ref_key][m] for idx in by_index])
            for m in METRIC_KEYS
        }

    return {
        "num_indices": len(by_index),
        "num_total_outputs": len(entries),
        "seeds_per_index": seeds_per_index,
        "per_index": per_index_avg,
        "aggregated": aggregated,
    }


# ---------------------------------------------------------------------------
# Summary upload
# ---------------------------------------------------------------------------

def _upload_json(data: dict, r2_key: str, r2: R2Client) -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        json.dump(data, tf, indent=2)
        tf_path = Path(tf.name)
    try:
        r2.upload_file(tf_path, r2_key)
        print(f"  [upload] {r2_key}")
    finally:
        tf_path.unlink(missing_ok=True)


def _build_and_upload_summaries(
    all_results: list[dict],
    nt_keys: list[tuple[str, str]],
    r2: R2Client,
) -> dict[tuple[str, str], dict]:
    """Build per-(n,t) summary JSONs and ablation_summary.json; upload all."""
    by_nt: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        by_nt[(r["n"], r["t"])].append(r)

    nt_stats: dict[tuple, dict] = {}
    for nt in nt_keys:
        n, t = nt
        stats = _compute_nt_stats(by_nt[nt])
        nt_stats[nt] = stats
        label = f"n_{n}_t_{t}"
        _upload_json(
            {"n": n, "t": t, **stats},
            f"{MODEL_PREFIX}/{label}/{label}_summary.json",
            r2,
        )

    ablation = {
        f"n_{n}_t_{t}": {
            "n": n, "t": t,
            "num_indices": nt_stats[(n, t)]["num_indices"],
            "num_total_outputs": nt_stats[(n, t)]["num_total_outputs"],
            "aggregated": nt_stats[(n, t)]["aggregated"],
        }
        for n, t in nt_keys
    }
    _upload_json(ablation, f"{MODEL_PREFIX}/ablation_summary.json", r2)

    return nt_stats


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _build_and_upload_plots(
    nt_keys: list[tuple[str, str]],
    nt_stats: dict[tuple[str, str], dict],
    r2: R2Client,
    local_plots_dir: Path | None = None,
) -> None:
    """Generate and upload PDF summary plots."""

    def _stat(nt: tuple, ref: str, metric: str) -> tuple[float | None, float | None]:
        agg = nt_stats[nt]["aggregated"][ref].get(metric, {})
        return agg.get("mean"), agg.get("se")

    means_gt = np.array([_stat(k, "gt",        "psnr_constrained")[0] or float("nan") for k in nt_keys])
    ses_gt   = np.array([_stat(k, "gt",        "psnr_constrained")[1] or 0.0           for k in nt_keys])
    means_proj = np.array([_stat(k, "projected", "psnr_constrained")[0] or float("nan") for k in nt_keys])
    ses_proj   = np.array([_stat(k, "projected", "psnr_constrained")[1] or 0.0           for k in nt_keys])

    valid_gt = ~np.isnan(means_gt)
    if not valid_gt.any():
        print("  [plots] no valid psnr_constrained (gt) data — skipping plots")
        return

    PSNR_THRESHOLD_DB = 0.5  # combos within this many dB of the best are colored

    best_idx = int(np.nanargmax(means_gt))
    best_mean = means_gt[best_idx]

    # Combos within PSNR_THRESHOLD_DB of the best (by gt PSNR)
    colored_indices = [i for i, m in enumerate(means_gt) if not np.isnan(m) and m >= best_mean - PSNR_THRESHOLD_DB]

    cmap = plt.get_cmap("tab10")
    color_map: dict[int, tuple] = {i: cmap(rank % 10) for rank, i in enumerate(colored_indices)}

    def _color(i: int) -> tuple:
        return color_map.get(i, (0.7, 0.7, 0.7, 1.0))

    labels = [_nt_label(n, t) for n, t in nt_keys]

    colored_proj_vals = {
        i: means_proj[i]
        for i in colored_indices
        if not np.isnan(means_proj[i])
    }
    highlight_idx = min(colored_proj_vals, key=colored_proj_vals.get) if colored_proj_vals else None

    # Shared y-axis limits within each plot type (gt plots share one range, projected
    # plots share another), so the 4 n-splits of each type are directly comparable.
    def _ylim(lo_vals: np.ndarray, hi_vals: np.ndarray) -> tuple[float, float]:
        y_lo = np.nanmin(lo_vals)
        y_hi = np.nanmax(hi_vals)
        y_margin = (y_hi - y_lo) * 0.05 or 1.0
        return (y_lo - y_margin, y_hi + y_margin)

    ylim_gt = _ylim(
        np.concatenate([means_gt - ses_gt, [best_mean - PSNR_THRESHOLD_DB]]),
        np.concatenate([means_gt + ses_gt, [best_mean]]),
    )
    ylim_proj = _ylim(means_proj - ses_proj, means_proj + ses_proj)

    # Split into one sub-plot per distinct n value, preserving global colors/best/highlight.
    n_values = sorted({n for n, _ in nt_keys}, key=lambda n: _nt_sort_key((n, "0"))[0])

    for n_val in n_values:
        group_indices = [i for i, (n, _t) in enumerate(nt_keys) if n == n_val]
        group_xs = np.arange(len(group_indices))
        group_labels = [labels[i] for i in group_indices]

        # ------------------------------------------------------------
        # Plot 1: psnr_constrained vs gt_image
        # ------------------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(max(8, len(group_indices) * 0.8 + 2), 5))

        best_color = _color(best_idx)
        ax1.axhspan(best_mean - PSNR_THRESHOLD_DB, best_mean, alpha=0.15, color=best_color, zorder=0)
        ax1.axhline(best_mean,                     color=best_color, linestyle="--", linewidth=1.2, zorder=1)
        ax1.axhline(best_mean - PSNR_THRESHOLD_DB, color=best_color, linestyle=":",  linewidth=0.8, zorder=1)

        for x, i in zip(group_xs, group_indices):
            m, se = means_gt[i], ses_gt[i]
            if np.isnan(m):
                continue
            ax1.errorbar(x, m, yerr=se, fmt="o", color=_color(i), capsize=4,
                         capthick=1.5, markersize=7, zorder=3)

        ax1.set_xticks(group_xs)
        ax1.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("PSNR constrained (dB)")
        ax1.set_title(f"n={n_val}")
        ax1.set_ylim(ylim_gt)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        fig1.tight_layout()

        # ------------------------------------------------------------
        # Plot 2: psnr_constrained vs projected_image (same x-order, same colors)
        # ------------------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(max(8, len(group_indices) * 0.8 + 2), 5))

        for x, i in zip(group_xs, group_indices):
            m, se = means_proj[i], ses_proj[i]
            if np.isnan(m):
                continue
            is_highlight = (i == highlight_idx)
            ax2.errorbar(
                x, m, yerr=se,
                fmt="D" if is_highlight else "o",
                color=_color(i),
                capsize=4, capthick=1.5,
                markersize=10 if is_highlight else 7,
                zorder=3,
                label=f"lowest proj PSNR among colored: {labels[i]}" if is_highlight else None,
            )

        if highlight_idx is not None and highlight_idx in group_indices:
            ax2.axvline(group_indices.index(highlight_idx), color=_color(highlight_idx),
                        linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)

        ax2.set_xticks(group_xs)
        ax2.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("PSNR constrained (dB)")
        ax2.set_title(f"n={n_val}")
        ax2.set_ylim(ylim_proj)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        if highlight_idx is not None and highlight_idx in group_indices:
            ax2.legend(fontsize=7, loc="lower left")
        fig2.tight_layout()

        # Upload both PDFs (and optionally save locally)
        for fig, fname in [(fig1, f"psnr_vs_gt_n{n_val}.pdf"), (fig2, f"psnr_vs_projected_n{n_val}.pdf")]:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                tf_path = Path(tf.name)
            try:
                fig.savefig(tf_path, format="pdf", bbox_inches="tight")
                r2_key = f"{MODEL_PREFIX}/{fname}"
                r2.upload_file(tf_path, r2_key)
                print(f"  [plots] uploaded {r2_key}")
                if local_plots_dir is not None:
                    local_plots_dir.mkdir(parents=True, exist_ok=True)
                    local_path = local_plots_dir / fname
                    shutil.copyfile(tf_path, local_path)
                    print(f"  [plots] saved {local_path}")
            finally:
                tf_path.unlink(missing_ok=True)
            plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FLUX inpainting results from R2")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip seeds whose per-seed JSON files already exist in R2",
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Skip metric computation; only regenerate summaries and plots from existing per-seed JSONs",
    )
    parser.add_argument(
        "--plots-only-aggregated", action="store_true",
        help=(
            "Skip metric computation and per-seed JSON loading; "
            "regenerate ablation_summary.json and plots directly from existing "
            "per-(n,t) n_<n>_t_<t>_summary.json files already in R2"
        ),
    )
    parser.add_argument(
        "--local-plots-dir", type=Path, default=None,
        help="If set, also save generated plot PDFs to this local directory",
    )
    return parser.parse_args()


def _load_nt_stats_from_summaries(r2: R2Client) -> tuple[list[tuple[str, str]], dict[tuple[str, str], dict]]:
    """
    Discover and load existing per-(n,t) summary JSONs from R2.
    Returns (nt_keys sorted, nt_stats dict).
    """
    all_keys = r2.list_keys(MODEL_PREFIX + "/")
    summary_re = re.compile(r"^(n_([\d.]+)_t_(.+))_summary\.json$")
    nt_stats: dict[tuple[str, str], dict] = {}

    for key in all_keys:
        fname = Path(key).name
        m = summary_re.match(fname)
        if not m:
            continue
        n, t = m.group(2), m.group(3)
        try:
            with tempfile.NamedTemporaryFile(suffix=".json") as tf:
                r2.download_file(key, Path(tf.name))
                data = json.loads(Path(tf.name).read_text())
            nt_stats[(n, t)] = data
            print(f"  [load] {key}")
        except Exception:
            print(f"  [load] failed to load {key}")

    nt_keys = sorted(nt_stats.keys(), key=lambda k: _nt_sort_key(k))
    return nt_keys, nt_stats


def main() -> None:
    args = parse_args()
    r2 = R2Client()

    if args.plots_only_aggregated:
        print("--plots-only-aggregated: loading per-(n,t) summary JSONs from R2...")
        nt_keys, nt_stats = _load_nt_stats_from_summaries(r2)
        if not nt_keys:
            print("No summary JSONs found — nothing to plot")
            return
        print(f"Loaded {len(nt_keys)} (n,t) combos")
        ablation = {
            f"n_{n}_t_{t}": {
                "n": n, "t": t,
                "num_indices": nt_stats[(n, t)].get("num_indices"),
                "num_total_outputs": nt_stats[(n, t)].get("num_total_outputs"),
                "aggregated": nt_stats[(n, t)].get("aggregated", {}),
            }
            for n, t in nt_keys
        }
        _upload_json(ablation, f"{MODEL_PREFIX}/ablation_summary.json", r2)
        print("Building and uploading plots...")
        _build_and_upload_plots(nt_keys, nt_stats, r2, local_plots_dir=args.local_plots_dir)
        print("\nDone.")
        return

    print(f"Listing seed images under {MODEL_PREFIX}/...")
    all_keys = r2.list_keys(MODEL_PREFIX + "/")
    seed_keys = [k for k in all_keys if SEED_RE.match(Path(k).name)]

    if not seed_keys:
        print(f"No seed_*.png files found under {MODEL_PREFIX}/")
        return

    print(f"Found {len(seed_keys)} seed images")

    # Parse keys → entries
    # key: blender/gt_geometry/<model_slug>/n_<n>_t_<t>/<index>/seed_<seed>.png
    entries: list[dict] = []
    for key in seed_keys:
        parts = key.split("/")
        if len(parts) < 6:
            continue
        nt_part = parts[-3]
        m = NT_RE.match(nt_part)
        if not m:
            continue
        n, t = m.group(1), m.group(2)
        try:
            index = int(parts[-2])
        except ValueError:
            continue
        seed_m = SEED_RE.match(parts[-1])
        if not seed_m:
            continue
        entries.append({
            "n": n, "t": t,
            "index": index,
            "seed": seed_m.group(1),
            "png_key": key,
            "prompt": "",
        })

    if not entries:
        print("Could not parse any valid entries from key list")
        return

    # Sorted (n, t) combos for consistent x-axis ordering
    all_nt = sorted({(e["n"], e["t"]) for e in entries}, key=_nt_sort_key)

    by_index: dict[int, list[dict]] = defaultdict(list)
    for e in entries:
        by_index[e["index"]].append(e)

    all_results: list[dict] = []

    if args.plots_only:
        print("--plots-only: loading existing metrics JSONs from R2...")
        for e in entries:
            gt_data, proj_data = {}, {}
            for suffix, store in [("_metrics_gt_image.json", "gt"), ("_metrics_projected_image.json", "projected")]:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".json") as tf:
                        r2.download_file(e["png_key"].replace(".png", suffix), Path(tf.name))
                        data = json.loads(Path(tf.name).read_text())
                    if store == "gt":
                        gt_data = data
                    else:
                        proj_data = data
                except Exception:
                    pass
            all_results.append({**e, "metrics_gt": gt_data, "metrics_projected": proj_data})
    else:
        with tempfile.TemporaryDirectory(prefix="eval_refs_") as tmp_str:
            tmp_root = Path(tmp_str)
            cache = _RefCache(r2, tmp_root)
            indices = sorted(by_index.keys())
            for idx_num, index in enumerate(indices):
                index_entries = by_index[index]
                print(f"[{idx_num+1}/{len(indices)}] index {index}: {len(index_entries)} seed(s)")
                refs = cache.get(index)
                if refs is None:
                    continue
                try:
                    results = _evaluate_index(
                        index, index_entries, refs, r2,
                        tmp_root / "seeds",
                        skip_existing=args.skip_existing,
                    )
                    all_results.extend(results)
                except Exception:
                    print(f"  [eval] index {index}: unexpected error")
                    traceback.print_exc()

    if not all_results:
        print("No results collected — skipping summaries and plots")
        return

    # Only include (n,t) combos that have results
    nt_with_results = {(r["n"], r["t"]) for r in all_results}
    nt_keys = [nt for nt in all_nt if nt in nt_with_results]

    print(f"\nBuilding summaries for {len(nt_keys)} (n, t) combos ({len(all_results)} total outputs)...")
    nt_stats = _build_and_upload_summaries(all_results, nt_keys, r2)

    print("Building and uploading plots...")
    _build_and_upload_plots(nt_keys, nt_stats, r2, local_plots_dir=args.local_plots_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
