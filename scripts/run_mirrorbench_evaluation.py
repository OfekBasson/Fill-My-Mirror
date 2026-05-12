"""
MirrorBench V2 evaluation script — compares multiple models on the full dataset.

Models evaluated (total dataset: indices 0–2910, 2911 images):

  Ours (estimated geometry):
    mirrorbench_v2/estimated_geometry/black-forest-labs--FLUX.1-Fill-dev/n_13.0_t_625.0/<idx>/seed_0.png
  FLUX.1 Fill (vanilla):
    mirrorbench_v2/estimated_geometry/black-forest-labs--FLUX.1-Fill-dev_vanilla/<idx>/seed_0.png
  FLUX.2 Klein (vanilla):
    mirrorbench_v2/estimated_geometry/black-forest-labs--FLUX.2-klein-base-9B_vanilla/<idx>/seed_0.png
  Qwen-2511 (vanilla):
    mirrorbench_v2/estimated_geometry/Qwen--Qwen-Image-Edit-2511/<idx>/seed_0.png
  Qwen (vanilla):
    mirrorbench_v2/estimated_geometry/Qwen--Qwen-Image-Edit/<idx>/seed_0.png
  MirrorVerse (GT geometry):
    mirrorbench_v2/mirrorfusion_depth_concat/gt_geometry/<idx>/seed_0.png
  Ours (GT geometry):
    mirrorbench_v2/gt_geometry/black-forest-labs--FLUX.1-Fill-dev/n_13.0_t_625.0/<idx>/seed_0.png

Reference masks are pulled from R2:
  Full mirror mask    : mirrorbench_v2/{geom_subdir}/<idx>/generative_refinement_mask.png
  Constrained mask    : mirrorbench_v2/{geom_subdir}/<idx>/constrained_pixels_gt_geometry_mask.png

Ground-truth images come from the local MirrorBench V2 dataset
(data/mirrorbench_v2/) via MirrorBenchV2SampleLoader.

Outputs (written to --output-dir and uploaded to R2 under mirrorbench_v2/evaluation/):
  missing_indices.txt           — per-model missing indices report
  <model_slug>/
    <index>_metrics.json        — per-index metrics
    summary.json                — per-model aggregated stats
  comparison_table.csv          — one row per model, one column per metric (mean)
  comparison_plots.pdf          — bar charts for each metric across models

Usage
-----
Evaluate all models (skipping per-index JSONs that already exist):
    python scripts/run_mirrorbench_evaluation.py --skip-existing

Regenerate summaries and plots from existing per-index JSONs:
    python scripts/run_mirrorbench_evaluation.py --plots-only

Specify a custom output directory:
    python scripts/run_mirrorbench_evaluation.py --output-dir outputs/mirrorbench_eval/
"""

from __future__ import annotations

import argparse
import json
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
from fill_my_mirror.loaders import MirrorBenchV2SampleLoader
from fill_my_mirror.storage import R2Client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL_INDICES = 2991  

R2_EVAL_PREFIX = "mirrorbench_v2/evaluation"

# All models are evaluated using the GT-geometry masks regardless of how they were generated
MASK_PREFIX = "mirrorbench_v2/gt_geometry"

METRIC_KEYS = (
    "psnr_full_mirror", "ssim_full_mirror", "lpips_full_mirror",
    "psnr_constrained", "ssim_constrained", "lpips_constrained",
)

MODELS: list[dict] = [
    {
        "name": "Ours (estimated geom.)",
        "slug": "ours_estimated",
        "key_template": "mirrorbench_v2/estimated_geometry/black-forest-labs--FLUX.1-Fill-dev/n_13.0_t_625.0/{index}/seed_0.png",
    },
    {
        "name": "FLUX.1 Fill",
        "slug": "flux1_fill_vanilla",
        "key_template": "mirrorbench_v2/estimated_geometry/black-forest-labs--FLUX.1-Fill-dev_vanilla/{index}/seed_0.png",
    },
    {
        "name": "FLUX.2 Klein",
        "slug": "flux2_klein_vanilla",
        "key_template": "mirrorbench_v2/estimated_geometry/black-forest-labs--FLUX.2-klein-base-9B_vanilla/{index}/seed_0.png",
    },
    {
        "name": "Qwen-2511",
        "slug": "qwen_2511_vanilla",
        "key_template": "mirrorbench_v2/estimated_geometry/Qwen--Qwen-Image-Edit-2511/{index}/seed_0.png",
    },
    {
        "name": "Qwen",
        "slug": "qwen_vanilla",
        "key_template": "mirrorbench_v2/estimated_geometry/Qwen--Qwen-Image-Edit/{index}/seed_0.png",
    },
    {
        "name": "MirrorVerse (GT geom.)",
        "slug": "mirrorfusion_gt",
        "key_template": "mirrorbench_v2/mirrorfusion_depth_concat/gt_geometry/{index}/seed_0.png",
    },
    {
        "name": "Ours (GT geom.)",
        "slug": "ours_gt",
        "key_template": "mirrorbench_v2/gt_geometry/black-forest-labs--FLUX.1-Fill-dev/n_13.0_t_625.0/{index}/seed_0.png",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r2_metrics_key(model_slug: str, index: int) -> str:
    return f"{R2_EVAL_PREFIX}/{model_slug}/{index}_metrics.json"


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
# Step 1 — Discover which indices exist per model
# ---------------------------------------------------------------------------

def discover_indices(r2: R2Client) -> dict[str, set[int]]:
    """Return {model_slug: set of available indices} by listing R2 keys."""
    present: dict[str, set[int]] = {m["slug"]: set() for m in MODELS}

    # Build a reverse map: prefix_dir -> [model_slug, ...]
    # We list R2 keys once per unique parent prefix (the part before {index})
    prefixes_to_models: dict[str, list[str]] = defaultdict(list)
    for m in MODELS:
        # key_template: "<prefix>/{index}/seed_0.png"
        parent = m["key_template"].split("/{index}/")[0]
        prefixes_to_models[parent].append(m["slug"])

    for prefix, slugs in prefixes_to_models.items():
        print(f"  Listing {prefix}/...")
        keys = r2.list_keys(prefix + "/")
        for key in keys:
            parts = key.split("/")
            # Find index segment: the component after the prefix depth
            prefix_depth = len(prefix.split("/"))
            if len(parts) <= prefix_depth:
                continue
            idx_str = parts[prefix_depth]
            if not idx_str.isdigit():
                continue
            if parts[-1] != "seed_0.png":
                continue
            idx = int(idx_str)
            for slug in slugs:
                # Verify the full key matches (in case multiple models share a prefix)
                expected = MODELS[next(i for i, m in enumerate(MODELS) if m["slug"] == slug)]["key_template"].format(index=idx)
                if key == expected:
                    present[slug].add(idx)

    return present


# ---------------------------------------------------------------------------
# Step 2 — Write missing-indices report
# ---------------------------------------------------------------------------

def write_missing_report(present: dict[str, set[int]], output_dir: Path) -> None:
    all_indices = set(range(TOTAL_INDICES))
    lines: list[str] = [f"MirrorBench V2 — missing indices report (total dataset: {TOTAL_INDICES} images)\n"]

    for m in MODELS:
        slug = m["slug"]
        found = present[slug]
        missing = sorted(all_indices - found)
        lines.append(f"\n{'=' * 70}")
        lines.append(f"Model : {m['name']}  ({slug})")
        lines.append(f"Found : {len(found)} / {TOTAL_INDICES}")
        lines.append(f"Missing ({len(missing)}): {missing if len(missing) <= 50 else str(missing[:50]) + ' ...'}")

    report = "\n".join(lines) + "\n"
    local_path = output_dir / "missing_indices.txt"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(report)
    print(f"  Saved {local_path}")


# ---------------------------------------------------------------------------
# Step 3 — Mask cache (per index, per geom subdir)
# ---------------------------------------------------------------------------

class _MaskCache:
    """Downloads and caches full-mirror + constrained masks from R2."""

    def __init__(self, r2: R2Client, tmp_root: Path):
        self._r2 = r2
        self._tmp_root = tmp_root
        self._cache: dict[tuple[str, int], dict | None] = {}

    def get(self, mask_prefix: str, index: int) -> dict | None:
        key = (mask_prefix, index)
        if key in self._cache:
            return self._cache[key]

        d = self._tmp_root / mask_prefix.replace("/", "_") / str(index)
        d.mkdir(parents=True, exist_ok=True)
        base = f"{mask_prefix}/{index}"
        masks: dict = {}
        ok = True

        for fname, dict_key in [
            ("generative_refinement_mask.png", "full_mirror_mask"),
            ("constrained_pixels_gt_geometry_mask.png", "constrained_mask"),
        ]:
            r2_key = f"{base}/{fname}"
            local = d / fname
            try:
                self._r2.download_file(r2_key, local)
                masks[dict_key] = Image.open(local)
            except Exception:
                print(f"    [mask] {mask_prefix}/{index}: cannot download {fname}")
                ok = False

        result = masks if ok else None
        self._cache[key] = result
        return result


# ---------------------------------------------------------------------------
# Step 4 — Per-index metric computation (all models at once)
# ---------------------------------------------------------------------------

def evaluate_index_all_models(
    active_models: list[dict],
    index: int,
    gt_image: Image.Image,
    masks: dict,
    r2: R2Client,
    tmp_root: Path,
    skip_existing: bool,
    output_dir: Path,
) -> dict[str, dict]:
    """
    Download generated images for all models at this index, run compute_metrics
    once (sharing gt_image and masks), then save per-model JSONs.

    Returns {model_slug: metrics_dict} for every model that succeeded.
    """
    # --- Determine which models still need computation ---
    pending: list[dict] = []
    cached: dict[str, dict] = {}

    for m in active_models:
        r2_key = _r2_metrics_key(m["slug"], index)
        if skip_existing and r2.key_exists(r2_key):
            try:
                with tempfile.NamedTemporaryFile(suffix=".json") as tf:
                    r2.download_file(r2_key, Path(tf.name))
                    cached[m["slug"]] = json.loads(Path(tf.name).read_text())
                continue
            except Exception:
                pass
        pending.append(m)

    if not pending:
        return cached

    # --- Download generated images for pending models ---
    img_dir = tmp_root / str(index)
    img_dir.mkdir(parents=True, exist_ok=True)

    gen_images: list[GeneratedImage] = []
    slug_to_local: dict[str, Path] = {}

    for m in pending:
        img_key = m["key_template"].format(index=index)
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
        return cached

    # --- Single compute_metrics call for all pending models ---
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
        print(f"    [eval] compute_metrics failed for index {index}")
        traceback.print_exc()
        for p in slug_to_local.values():
            p.unlink(missing_ok=True)
        return cached

    # --- Extract per-model rows from the DataFrame and persist ---
    results = dict(cached)
    for _, row in df.iterrows():
        slug = row["name"]
        metrics = row.drop("name").to_dict()
        r2_key = _r2_metrics_key(slug, index)
        local_json = output_dir / slug / f"{index}_metrics.json"
        _save_json_local(metrics, local_json)
        _upload_json(metrics, r2_key, r2)
        results[slug] = metrics

    for p in slug_to_local.values():
        p.unlink(missing_ok=True)

    return results


# ---------------------------------------------------------------------------
# Step 5 — Aggregate statistics
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


def build_summary(model: dict, index_metrics: dict[int, dict]) -> dict:
    """Build per-metric aggregated stats for one model."""
    per_metric: dict[str, list[float]] = {m: [] for m in METRIC_KEYS}
    for metrics in index_metrics.values():
        for m in METRIC_KEYS:
            v = metrics.get(m)
            if v is not None:
                per_metric[m].append(v)
    return {
        "model_name": model["name"],
        "model_slug": model["slug"],
        "num_indices_evaluated": len(index_metrics),
        "aggregated": {m: _agg_stats(vals) for m, vals in per_metric.items()},
    }


# ---------------------------------------------------------------------------
# Step 6 — Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(summaries: list[dict]) -> pd.DataFrame:
    rows = []
    for s in summaries:
        row = {"model": s["model_name"], "n": s["num_indices_evaluated"]}
        for m in METRIC_KEYS:
            agg = s["aggregated"].get(m, {})
            row[f"{m}_mean"] = agg.get("mean")
            row[f"{m}_se"]   = agg.get("se")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 7 — Plots
# ---------------------------------------------------------------------------

def build_plots(summaries: list[dict], output_dir: Path, r2: R2Client) -> None:
    model_names = [s["model_name"] for s in summaries]
    xs = np.arange(len(model_names))
    width = 0.6

    plot_metrics = [
        ("psnr_full_mirror",   "PSNR full mirror (dB)"),
        ("psnr_constrained",   "PSNR constrained (dB)"),
        ("ssim_full_mirror",   "SSIM full mirror"),
        ("ssim_constrained",   "SSIM constrained"),
        ("lpips_full_mirror",  "LPIPS full mirror ↓"),
        ("lpips_constrained",  "LPIPS constrained ↓"),
    ]

    n_metrics = len(plot_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(max(10, len(model_names) * 1.4 + 2), 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(model_names))]

    for ax, (metric, ylabel) in zip(axes, plot_metrics):
        means = np.array([s["aggregated"].get(metric, {}).get("mean") or float("nan") for s in summaries])
        ses   = np.array([s["aggregated"].get(metric, {}).get("se")   or 0.0           for s in summaries])

        bars = ax.bar(xs, np.where(np.isnan(means), 0, means), width, color=colors, alpha=0.85, zorder=3)
        ax.errorbar(xs, np.where(np.isnan(means), 0, means), yerr=ses,
                    fmt="none", color="black", capsize=4, capthick=1.2, zorder=4)

        # Annotate missing (NaN) bars
        for i, m in enumerate(means):
            if np.isnan(m):
                ax.text(xs[i], 0.01, "N/A", ha="center", va="bottom", fontsize=7, color="gray",
                        transform=ax.get_xaxis_transform())

        ax.set_xticks(xs)
        ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.set_xlim(-0.5, len(model_names) - 0.5)

    fig.suptitle("MirrorBench V2 — Model Comparison", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    local_pdf = output_dir / "comparison_plots.pdf"
    fig.savefig(local_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {local_pdf}")

    r2_key = f"{R2_EVAL_PREFIX}/comparison_plots.pdf"
    r2.upload_file(local_pdf, r2_key)
    print(f"  Uploaded {r2_key}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple models on MirrorBench V2")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mirrorbench_eval"),
                        help="Local directory for outputs (default: outputs/mirrorbench_eval/)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip indices whose per-index JSON already exists in R2")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip metric computation; reload existing per-index JSONs from R2")
    parser.add_argument("--models", nargs="+", metavar="SLUG",
                        help="Restrict evaluation to specific model slugs (default: all)")
    return parser.parse_args()


def _load_existing_metrics(model: dict, present: set[int], r2: R2Client) -> dict[int, dict]:
    """Load per-index metric JSONs from R2 for one model."""
    result: dict[int, dict] = {}
    for index in sorted(present):
        r2_key = _r2_metrics_key(model["slug"], index)
        try:
            with tempfile.NamedTemporaryFile(suffix=".json") as tf:
                r2.download_file(r2_key, Path(tf.name))
                result[index] = json.loads(Path(tf.name).read_text())
        except Exception:
            pass
    return result


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    active_models = MODELS
    if args.models:
        active_models = [m for m in MODELS if m["slug"] in args.models]
        if not active_models:
            print(f"No models matched slugs: {args.models}")
            return

    r2 = R2Client()

    # --- Discover present indices ---
    print("Discovering available indices in R2...")
    present = discover_indices(r2)

    # --- Missing-indices report ---
    print("\nWriting missing-indices report...")
    write_missing_report(present, output_dir)

    # --- Load summaries for each model ---
    all_summaries: list[dict] = []

    if args.plots_only:
        print("\n--plots-only: loading existing per-index metrics from R2...")
        for m in active_models:
            slug = m["slug"]
            print(f"\n  Loading {m['name']} ({len(present[slug])} indices)...")
            index_metrics = _load_existing_metrics(m, present[slug], r2)
            summary = build_summary(m, index_metrics)
            _save_json_local(summary, output_dir / slug / "summary.json")
            _upload_json(summary, f"{R2_EVAL_PREFIX}/{slug}/summary.json", r2)
            all_summaries.append(summary)
    else:
        print("\nLoading MirrorBench V2 dataset (HuggingFace)...")
        loader = MirrorBenchV2SampleLoader()
        print(f"  Dataset has {len(loader)} samples")

        # Collect all indices that appear in at least one active model
        all_active_indices = sorted({idx for m in active_models for idx in present[m["slug"]]})
        print(f"  {len(all_active_indices)} unique indices across active models")

        # Per-model accumulator: slug -> {index -> metrics}
        index_metrics_by_model: dict[str, dict[int, dict]] = {m["slug"]: {} for m in active_models}

        with tempfile.TemporaryDirectory(prefix="mirrorbench_eval_") as tmp_str:
            tmp_root = Path(tmp_str)
            mask_cache = _MaskCache(r2, tmp_root / "masks")

            for i, index in enumerate(all_active_indices):
                if index >= len(loader):
                    print(f"  [{i+1}/{len(all_active_indices)}] index {index}: out of dataset range — skipping")
                    continue

                masks = mask_cache.get(MASK_PREFIX, index)
                if masks is None:
                    print(f"  [{i+1}/{len(all_active_indices)}] index {index}: masks unavailable — skipping")
                    continue

                try:
                    sample = loader.load(index)
                    gt_image = Image.open(sample.gt_image_path).convert("RGB")
                except Exception:
                    print(f"  [{i+1}/{len(all_active_indices)}] index {index}: failed to load GT — skipping")
                    traceback.print_exc()
                    continue

                # Only pass models that have this index
                models_for_index = [m for m in active_models if index in present[m["slug"]]]

                results = evaluate_index_all_models(
                    models_for_index, index, gt_image, masks, r2,
                    tmp_root / "imgs",
                    skip_existing=args.skip_existing,
                    output_dir=output_dir,
                )
                for slug, metrics in results.items():
                    index_metrics_by_model[slug][index] = metrics

                if (i + 1) % 100 == 0 or (i + 1) == len(all_active_indices):
                    counts = ", ".join(f"{m['slug']}:{len(index_metrics_by_model[m['slug']])}" for m in active_models)
                    print(f"  [{i+1}/{len(all_active_indices)}] evaluated — {counts}")

        for m in active_models:
            summary = build_summary(m, index_metrics_by_model[m["slug"]])
            _save_json_local(summary, output_dir / m["slug"] / "summary.json")
            _upload_json(summary, f"{R2_EVAL_PREFIX}/{m['slug']}/summary.json", r2)
            all_summaries.append(summary)
            print(f"  Summary saved for {m['name']}")

    if not all_summaries:
        print("No summaries built — nothing to plot")
        return

    # --- Comparison table ---
    print("\nBuilding comparison table...")
    table = build_comparison_table(all_summaries)
    table_path = output_dir / "comparison_table.csv"
    table.to_csv(table_path, index=False, float_format="%.4f")
    print(f"  Saved {table_path}")
    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.4f}" if x is not None else "N/A"))

    r2.upload_file(table_path, f"{R2_EVAL_PREFIX}/comparison_table.csv")
    print(f"  Uploaded {R2_EVAL_PREFIX}/comparison_table.csv")

    # --- Plots ---
    print("\nBuilding plots...")
    build_plots(all_summaries, output_dir, r2)

    print("\nDone.")


if __name__ == "__main__":
    main()
