"""
Statistical significance tests and confidence intervals for real-image evaluation results.

Loads per-(index, seed) metric JSONs from outputs/real_eval/, aggregates per-index means
across seeds, then runs pairwise tests between a reference model and all others.

Tests used
----------
- Paired t-test (parametric): assumes the per-index differences are approximately normal.
  Valid when n ≥ 30 or the distribution is roughly symmetric. Reports p-value and 95% CI
  on the mean difference.
- Wilcoxon signed-rank test (non-parametric): makes no normality assumption. More robust
  for small or skewed samples.

Both tests are paired: for each image index, we compare the same index across two models,
so individual image difficulty cancels out (analogous to a within-subject design).

Multiple comparisons
--------------------
Benjamini–Hochberg correction is applied across baseline comparisons separately for each
metric. For example, for psnr_constrained the t-test p-values from all baseline-vs-ref
pairs are corrected together, and similarly for Wilcoxon p-values.

Confidence intervals
--------------------
Bootstrap 95% CI on the mean difference is computed via 10,000 resamples of the
per-index difference vector. This is distribution-free and reliable even for small n.

Sign convention
---------------
All differences are oriented so that a positive value means the other model is better
than the reference (for PSNR/SSIM: other − ref; for LPIPS: ref − other, since lower is
better). The field is named mean_diff_positive_other_better to make this explicit.

Usage
-----
    python scripts/stats_significance.py
    python scripts/stats_significance.py --ref ours_interp --metrics psnr_constrained ssim_constrained
    python scripts/stats_significance.py --output-dir outputs/real_eval --alpha 0.01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

METRIC_KEYS = [
    "psnr_full_mirror",
    "ssim_full_mirror",
    "lpips_full_mirror",
    "psnr_constrained",
    "ssim_constrained",
    "lpips_constrained",
]

MODEL_SLUGS = [
    "ours_interp",
    "ours_no_interp",
    "flux1_fill_vanilla",
    "qwen_2511_vanilla",
    "mirrorfusion",
]

SEEDS = [0, 42, 512]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_per_index_means(output_dir: Path, slug: str, seeds: list[int]) -> dict[int, dict[str, float]]:
    """
    For each image index, load all available seed JSONs and average metrics across seeds.
    Returns {index: {metric: mean_value}}.
    """
    model_dir = output_dir / slug
    if not model_dir.exists():
        return {}

    index_data: dict[int, dict[str, list[float]]] = {}
    for json_file in sorted(model_dir.glob("*_seed_*_metrics.json")):
        stem = json_file.stem  # e.g. "3_seed_42_metrics"
        parts = stem.split("_seed_")
        if len(parts) != 2:
            continue
        try:
            index = int(parts[0])
            seed = int(parts[1].replace("_metrics", ""))
        except ValueError:
            continue
        if seed not in seeds:
            continue
        data = json.loads(json_file.read_text())
        if index not in index_data:
            index_data[index] = {m: [] for m in METRIC_KEYS}
        for m in METRIC_KEYS:
            v = data.get(m)
            if v is not None:
                index_data[index][m].append(v)

    return {
        idx: {m: float(np.mean(vals)) for m, vals in mdict.items() if vals}
        for idx, mdict in index_data.items()
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap percentile CI on the mean of diffs."""
    rng = np.random.default_rng(0)
    means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def bh_correct(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg FDR correction. Returns adjusted p-values in input order."""
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    sorted_p = np.array(pvals)[order]
    adjusted = np.minimum(1.0, sorted_p * n / (np.arange(n) + 1))
    # Enforce monotonicity (cumulative min from the right)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    result = np.empty(n)
    result[order] = adjusted
    return result.tolist()


def compare_models(
    ref_data: dict[int, dict[str, float]],
    other_data: dict[int, dict[str, float]],
    metric: str,
    alpha: float,
) -> dict:
    """
    Paired comparison of other vs ref on a single metric over shared indices.

    Returns raw p-values; BH correction is applied by the caller across all baselines.
    mean_diff_positive_other_better: positive = other model is better than ref,
    regardless of metric direction (sign is flipped for LPIPS).
    """
    shared = sorted(set(ref_data) & set(other_data))
    valid = [i for i in shared if metric in ref_data[i] and metric in other_data[i]]
    ref_vals = np.array([ref_data[i][metric] for i in valid])
    oth_vals = np.array([other_data[i][metric] for i in valid])

    if len(ref_vals) < 3:
        return {"n": len(ref_vals), "error": "too few paired samples"}

    flip = metric.startswith("lpips")
    # positive diff = other is better than ref
    diffs = (ref_vals - oth_vals) if flip else (oth_vals - ref_vals)

    mean_diff = float(np.mean(diffs))
    t_stat, t_pval = stats.ttest_rel(ref_vals, oth_vals) if flip else stats.ttest_rel(oth_vals, ref_vals)

    if np.allclose(diffs, 0):
        w_stat, w_pval = np.nan, 1.0
    else:
        w_stat, w_pval = stats.wilcoxon(diffs, alternative="two-sided")

    ci_lo, ci_hi = bootstrap_ci(diffs, alpha=alpha)

    return {
        "n_paired": len(ref_vals),
        "mean_ref": float(np.mean(ref_vals)),
        "mean_other": float(np.mean(oth_vals)),
        "mean_diff_positive_other_better": mean_diff,
        "bootstrap_ci": [ci_lo, ci_hi],
        "ttest_pval": float(t_pval),
        "ttest_stat": float(t_stat),
        "wilcoxon_pval": float(w_pval),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise significance tests vs. a reference model")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/real_eval"))
    p.add_argument("--ref", default="ours_interp", help="Reference model slug")
    p.add_argument("--models", nargs="+", default=None, help="Other slugs to compare (default: all)")
    p.add_argument("--metrics", nargs="+", default=None, help="Metrics to test (default: all)")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    p.add_argument("--save", type=Path, default=None,
                   help="Path to save results as JSON (default: <output-dir>/significance_results.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = sorted(set(args.seeds))
    metrics = args.metrics or METRIC_KEYS
    other_slugs = [s for s in (args.models or MODEL_SLUGS) if s != args.ref]

    print(f"Loading reference model: {args.ref}")
    ref_data = load_per_index_means(args.output_dir, args.ref, seeds)
    print(f"  {len(ref_data)} indices loaded")

    all_model_data = {slug: load_per_index_means(args.output_dir, slug, seeds) for slug in other_slugs}

    # Collect raw results first, then apply BH per metric
    # results[metric][slug] = compare_models output dict (or error)
    results: dict[str, dict[str, dict]] = {}
    for metric in metrics:
        results[metric] = {}
        for slug in other_slugs:
            results[metric][slug] = compare_models(ref_data, all_model_data[slug], metric, args.alpha)

        # BH correction across baselines, separately for t-test and Wilcoxon
        valid_slugs = [s for s in other_slugs if "error" not in results[metric][s]]
        if valid_slugs:
            t_pvals = [results[metric][s]["ttest_pval"] for s in valid_slugs]
            w_pvals = [results[metric][s]["wilcoxon_pval"] for s in valid_slugs]
            t_adj = bh_correct(t_pvals)
            w_adj = bh_correct(w_pvals)
            for slug, tp, wp in zip(valid_slugs, t_adj, w_adj):
                results[metric][slug]["ttest_pval_bh"] = tp
                results[metric][slug]["wilcoxon_pval_bh"] = wp
                results[metric][slug]["significant_ttest_bh"] = bool(tp < args.alpha)
                results[metric][slug]["significant_wilcoxon_bh"] = bool(wp < args.alpha)

    # Save
    save_path = args.save or args.output_dir / "significance_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {save_path}")

    # Print
    for slug in other_slugs:
        print(f"\n{'=' * 80}")
        print(f"  {slug}  vs.  {args.ref}  (α={args.alpha}, BH-corrected)")
        print(f"{'=' * 80}")
        for metric in metrics:
            res = results[metric][slug]
            if "error" in res:
                print(f"  {metric:30s}  [SKIP: {res['error']}]")
                continue
            diff = res["mean_diff_positive_other_better"]
            ci = res["bootstrap_ci"]
            direction = "other better" if diff > 0 else "ref  better"
            sig_t  = "✓" if res.get("significant_ttest_bh")     else "✗"
            sig_w  = "✓" if res.get("significant_wilcoxon_bh")  else "✗"
            t_raw  = res["ttest_pval"]
            t_adj  = res.get("ttest_pval_bh", float("nan"))
            w_raw  = res["wilcoxon_pval"]
            w_adj  = res.get("wilcoxon_pval_bh", float("nan"))
            print(
                f"  {metric:30s}  "
                f"Δ={diff:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
                f"t p={t_raw:.4f} (BH {t_adj:.4f}){sig_t}  "
                f"W p={w_raw:.4f} (BH {w_adj:.4f}){sig_w}  "
                f"({direction}, n={res['n_paired']})"
            )


if __name__ == "__main__":
    main()
