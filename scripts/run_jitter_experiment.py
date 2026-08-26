"""
Full jitter-PSNR experiment: computes compute_jitter_psnr (a small per-pixel
local-offset search) for every (model, index, seed) already cached locally
under outputs/shift_tolerant_psnr/shift_tolerant_psnr_v1/, on both the
full-mirror and constrained masks.

This is exploratory: see the compute_jitter_psnr docstring in
fill_my_mirror/evaluation/metrics_computation.py for why this metric lacks a
coherence constraint across pixels and is likely partly reward-hacked by
local texture self-similarity, not purely measuring real content fidelity.

Purely CPU/numpy — no GPU, no LPIPS/SSIM, no R2. Reads locally cached
generated/ground_truth/mask PNGs already downloaded by an earlier pipeline
run, so no network access is needed here.

Usage
-----
    python scripts/run_jitter_experiment.py --dataset blender
    python scripts/run_jitter_experiment.py --dataset mirrorbench_v2 --workers 16
    python scripts/run_jitter_experiment.py --dataset real --workers 16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fill_my_mirror.evaluation.metrics_computation import (  # noqa: E402
    _pil_to_rgb01,
    _prepare_mask,
    _resize_rgb01,
    compute_jitter_psnr,
)

SOURCE_ROOT = Path("outputs/shift_tolerant_psnr/shift_tolerant_psnr_v1")
OUTPUT_ROOT = Path("outputs/jitter_psnr_v1")

DATASET_GLOBS = {
    "blender": ["blender"],
    "mirrorbench_v2": sorted(p.name for p in SOURCE_ROOT.glob("mirrorbench_v2_start*") if p.is_dir()),
    "real": sorted(p.name for p in SOURCE_ROOT.glob("real_start*") if p.is_dir())
            + (["real"] if (SOURCE_ROOT / "real").is_dir() else []),
}


def _find_seed_dirs(dataset: str) -> list[Path]:
    dirs = []
    for shard_name in DATASET_GLOBS[dataset]:
        shard = SOURCE_ROOT / shard_name
        for gen_png in shard.glob("*/*/seed_*/generated.png"):
            dirs.append(gen_png.parent)
    return dirs


def _process_one(seed_dir: Path, jitter_radius: int) -> dict | None:
    try:
        gt = Image.open(seed_dir / "ground_truth.png").convert("RGB")
        gen = Image.open(seed_dir / "generated.png").convert("RGB")
        full_mask_pil = Image.open(seed_dir / "mirror_mask.png").convert("L")
        constrained_mask_pil = Image.open(seed_dir / "constrained_mask.png").convert("L")
    except Exception:
        return None

    ref = _pil_to_rgb01(gt)
    arr = _pil_to_rgb01(gen)
    if arr.shape[:2] != ref.shape[:2]:
        arr = _resize_rgb01(arr, (ref.shape[0], ref.shape[1]))
    full_mask = _prepare_mask(full_mask_pil, ref.shape[:2])
    constrained_mask = _prepare_mask(constrained_mask_pil, ref.shape[:2])
    if full_mask.sum() == 0:
        return None

    model_slug = seed_dir.parent.parent.name
    index = int(seed_dir.parent.name)
    seed = int(seed_dir.name.replace("seed_", ""))

    row: dict = {"model_slug": model_slug, "index": index, "seed": seed}

    r_full = compute_jitter_psnr(ref, arr, full_mask, jitter_radius=jitter_radius)
    row["psnr_plain_full_mirror"] = r_full["psnr_plain"]
    row["psnr_jitter_full_mirror"] = r_full["psnr_jitter"]
    row["inflation_full_mirror"] = r_full["psnr_jitter"] - r_full["psnr_plain"]

    if constrained_mask.sum() > 0:
        r_constr = compute_jitter_psnr(ref, arr, constrained_mask, jitter_radius=jitter_radius)
        row["psnr_plain_constrained"] = r_constr["psnr_plain"]
        row["psnr_jitter_constrained"] = r_constr["psnr_jitter"]
        row["inflation_constrained"] = r_constr["psnr_jitter"] - r_constr["psnr_plain"]

    return row


class _StarArgs:
    """Picklable callable binding jitter_radius for Pool.imap_unordered."""

    def __init__(self, jitter_radius: int):
        self.jitter_radius = jitter_radius

    def __call__(self, seed_dir: Path) -> dict | None:
        return _process_one(seed_dir, self.jitter_radius)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_GLOBS.keys()))
    parser.add_argument("--jitter-radius", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = args.output or OUTPUT_ROOT / f"{args.dataset}_jitter_rows.csv"

    seed_dirs = _find_seed_dirs(args.dataset)
    print(f"[{args.dataset}] {len(seed_dirs)} seed dirs found locally.")
    print(f"jitter_radius={args.jitter_radius} workers={args.workers}")

    t0 = time.time()
    rows: list[dict] = []
    if args.workers <= 1:
        for i, d in enumerate(seed_dirs):
            r = _process_one(d, args.jitter_radius)
            if r is not None:
                rows.append(r)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  ...{i + 1}/{len(seed_dirs)} ({elapsed:.0f}s elapsed, {elapsed / (i + 1):.3f}s/row)", flush=True)
    else:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=lambda: os.environ.update(
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        )) as pool:
            for i, r in enumerate(pool.imap_unordered(_StarArgs(args.jitter_radius), seed_dirs, chunksize=16)):
                if r is not None:
                    rows.append(r)
                if (i + 1) % 500 == 0:
                    elapsed = time.time() - t0
                    print(f"  ...{i + 1}/{len(seed_dirs)} ({elapsed:.0f}s elapsed, {elapsed / (i + 1):.3f}s/row)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n[{args.dataset}] wrote {len(df)} rows -> {out_path} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
