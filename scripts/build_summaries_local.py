"""Build summaries, comparison table, and plots from local per-index metric JSONs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL_DIR = Path("outputs/mirrorbench_eval")

METRIC_KEYS = (
    "psnr_full_mirror", "ssim_full_mirror", "lpips_full_mirror",
    "psnr_constrained", "ssim_constrained", "lpips_constrained",
    "clip_similarity",
)

MODELS = [
    {"name": "Ours (estimated geom.)", "slug": "ours_estimated"},
    {"name": "FLUX.1 Fill",            "slug": "flux1_fill_vanilla"},
    {"name": "FLUX.2 Klein",           "slug": "flux2_klein_vanilla"},
    {"name": "Qwen-2511",              "slug": "qwen_2511_vanilla"},
    {"name": "Qwen",                   "slug": "qwen_vanilla"},
    {"name": "MirrorVerse (GT geom.)", "slug": "mirrorfusion_gt"},
    {"name": "Ours (GT geom.)",        "slug": "ours_gt"},
]


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


def load_model_metrics(slug: str) -> dict[int, dict]:
    model_dir = EVAL_DIR / slug
    result = {}
    for f in sorted(model_dir.glob("*_metrics.json")):
        idx = int(f.stem.replace("_metrics", ""))
        result[idx] = json.loads(f.read_text())
    return result


def build_summary(model: dict, index_metrics: dict[int, dict]) -> dict:
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


def build_plots(summaries: list[dict]) -> None:
    model_names = [s["model_name"] for s in summaries]
    xs = np.arange(len(model_names))
    width = 0.6

    plot_metrics = [
        ("psnr_full_mirror",  "PSNR full mirror (dB)",  "psnr"),
        ("psnr_constrained",  "PSNR constrained (dB)",  "psnr"),
        ("ssim_full_mirror",  "SSIM full mirror",        "ssim"),
        ("ssim_constrained",  "SSIM constrained",        "ssim"),
        ("lpips_full_mirror", "LPIPS full mirror ↓",    "lpips"),
        ("lpips_constrained", "LPIPS constrained ↓",    "lpips"),
    ]

    # Compute shared y-axis limits per metric group
    def _range(keys):
        vals = [s["aggregated"].get(k, {}).get("mean") for s in summaries for k in keys if s["aggregated"].get(k, {}).get("mean") is not None]
        stds = [s["aggregated"].get(k, {}).get("std") or 0.0 for s in summaries for k in keys if s["aggregated"].get(k, {}).get("mean") is not None]
        lo = min(v - d for v, d in zip(vals, stds))
        hi = max(v + d for v, d in zip(vals, stds))
        pad = (hi - lo) * 0.1
        return max(0, lo - pad), hi + pad

    psnr_keys  = ["psnr_full_mirror",  "psnr_constrained"]
    ssim_keys  = ["ssim_full_mirror",  "ssim_constrained"]
    lpips_keys = ["lpips_full_mirror", "lpips_constrained"]
    ylims = {
        "psnr":  _range(psnr_keys),
        "ssim":  _range(ssim_keys),
        "lpips": _range(lpips_keys),
    }

    fig, axes = plt.subplots(len(plot_metrics), 1,
                             figsize=(max(10, len(model_names) * 1.4 + 2), 4 * len(plot_metrics)))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(model_names))]

    for ax, (metric, ylabel, group) in zip(axes, plot_metrics):
        means = np.array([s["aggregated"].get(metric, {}).get("mean") or float("nan") for s in summaries])
        stds  = np.array([s["aggregated"].get(metric, {}).get("std")  or 0.0           for s in summaries])
        ses   = np.array([s["aggregated"].get(metric, {}).get("se")   or 0.0           for s in summaries])

        # Sort by mean ascending (NaN last)
        order = np.argsort(np.where(np.isnan(means), np.inf, means))
        sorted_names  = [model_names[i] for i in order]
        sorted_means  = means[order]
        sorted_stds   = stds[order]
        sorted_ses    = ses[order]
        sorted_colors = [colors[i] for i in order]

        bar_means = np.where(np.isnan(sorted_means), 0, sorted_means)
        bars = ax.bar(xs, bar_means, width, color=sorted_colors, alpha=0.85, zorder=3)

        # std as shaded range on top of bars, se as error caps
        for xi, (mean_v, std_v, se_v) in enumerate(zip(sorted_means, sorted_stds, sorted_ses)):
            if np.isnan(mean_v):
                ax.text(xi, 0.01, "N/A", ha="center", va="bottom", fontsize=7, color="gray",
                        transform=ax.get_xaxis_transform())
                continue
            ax.fill_between([xi - width/2, xi + width/2],
                            mean_v - std_v, mean_v + std_v,
                            color="black", alpha=0.12, zorder=4)
            ax.errorbar(xi, mean_v, yerr=se_v,
                        fmt="none", color="black", capsize=5, capthick=1.5, zorder=5)

        ax.set_xticks(xs)
        ax.set_xticklabels(sorted_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(*ylims[group])
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.set_xlim(-0.5, len(model_names) - 0.5)

    fig.suptitle("MirrorBench V2 — Model Comparison", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = EVAL_DIR / "comparison_plots.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    all_summaries = []
    for m in MODELS:
        print(f"Loading {m['name']}...")
        index_metrics = load_model_metrics(m["slug"])
        summary = build_summary(m, index_metrics)
        out = EVAL_DIR / m["slug"] / "summary.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"  {summary['num_indices_evaluated']} indices — saved {out}")
        all_summaries.append(summary)

    print("\nBuilding comparison table...")
    table = build_comparison_table(all_summaries)
    table_path = EVAL_DIR / "comparison_table.csv"
    table.to_csv(table_path, index=False, float_format="%.4f")
    print(f"Saved {table_path}")
    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.4f}" if x is not None else "N/A"))

    print("\nBuilding plots...")
    build_plots(all_summaries)
    print("\nDone.")


if __name__ == "__main__":
    main()
