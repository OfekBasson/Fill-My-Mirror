"""
Merge sharded RCS evaluation results
=====================================
Combines metrics.csv from multiple shard output-dirs (produced by running
evaluate_rcs_on_dataset_with_known_gt_masks.py in parallel across GPUs, each
with a disjoint --indices subset and its own --output-dir) into one combined
metrics.csv/ranking.csv/scatter plot.

Usage
-----
    conda run -n fill-my-mirror python scripts/merge_rcs_shards.py \
        --shard-dirs outputs/rcs_mirrorbench_eval/shard_0 \
                     outputs/rcs_mirrorbench_eval/shard_1 \
                     outputs/rcs_mirrorbench_eval/shard_2 \
                     outputs/rcs_mirrorbench_eval/shard_3 \
        --output-dir outputs/rcs_mirrorbench_eval/combined
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded RCS evaluation metrics.csv files.")
    parser.add_argument("--shard-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for shard_dir in args.shard_dirs:
        metrics_path = Path(shard_dir) / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"{metrics_path} not found.")
        dfs.append(pd.read_csv(metrics_path))

    df = pd.concat(dfs, ignore_index=True)
    duplicated = df["idx"].duplicated()
    if duplicated.any():
        raise ValueError(f"Duplicate idx found across shards: {sorted(df.loc[duplicated, 'idx'].tolist())}")

    df = df.sort_values("idx").reset_index(drop=True)
    df.to_csv(output_dir / "metrics.csv", index=False)

    ranking = df.sort_values("f1", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    ranking.to_csv(output_dir / "ranking.csv", index=False)

    mean_p, mean_r, mean_f1 = df["precision"].mean(), df["recall"].mean(), df["f1"].mean()

    print("\n" + "=" * 60)
    print(f"Merged RCS Evaluation  ({len(args.shard_dirs)} shards)")
    print(f"  Samples evaluated : {len(df)}")
    print(f"  Mean Precision    : {mean_p:.4f}")
    print(f"  Mean Recall       : {mean_r:.4f}")
    print(f"  Mean F1           : {mean_f1:.4f}")
    print("=" * 60)
    print("\nTop 10 (best F1):")
    print(ranking.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBottom 10 (worst F1):")
    print(ranking.tail(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["recall"], df["precision"], alpha=0.5, s=15, color="steelblue")
    ax.axvline(mean_r, color="red", linestyle="--", linewidth=1, label=f"mean R={mean_r:.3f}")
    ax.axhline(mean_p, color="orange", linestyle="--", linewidth=1, label=f"mean P={mean_p:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(f"RCS  (merged, {len(df)} samples)", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_scatter.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved merged results to {output_dir}")


if __name__ == "__main__":
    main()
