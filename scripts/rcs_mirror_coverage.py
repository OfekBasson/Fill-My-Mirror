"""
Compute how much of the mirror region is covered by the RCS mask.

For each sample_<idx>/ that has rcs_mask_redilated.png, loads the mirror mask
from the MirrorBench HDF5 dataset and computes:

    coverage = |rcs_mask ∩ mirror_mask| / |mirror_mask|

Saves coverage.txt inside each sample dir, plus a summary CSV and txt.

Usage
-----
    python scripts/rcs_mirror_coverage.py
    python scripts/rcs_mirror_coverage.py --eval-dir outputs/rcs_mirrorbench_eval
    python scripts/rcs_mirror_coverage.py --indices 0 5 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fill_my_mirror.loaders import MirrorBenchV2SampleLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def process_sample(idx: int, sample_dir: Path, loader: MirrorBenchV2SampleLoader) -> dict | None:
    rcs_path = sample_dir / "rcs_mask_redilated.png"
    if not rcs_path.exists():
        logger.warning("[%d] rcs_mask_redilated.png not found — skipping.", idx)
        return None

    try:
        sample = loader.load(idx)
        mirror_mask = _load_mask(Path(sample.mask_path))
    except Exception as e:
        logger.warning("[%d] Cannot load mirror mask: %s", idx, e)
        return None

    rcs_mask = _load_mask(rcs_path)

    # Resize rcs_mask to mirror_mask resolution if they differ
    if rcs_mask.shape != mirror_mask.shape:
        H, W = mirror_mask.shape
        pil = Image.fromarray(rcs_mask.astype(np.uint8) * 255, mode="L")
        rcs_mask = np.asarray(pil.resize((W, H), Image.NEAREST)) > 127

    n_mirror = int(mirror_mask.sum())
    if n_mirror == 0:
        logger.warning("[%d] Mirror mask is empty — skipping.", idx)
        return None

    n_covered = int((rcs_mask & mirror_mask).sum())
    coverage = n_covered / n_mirror

    (sample_dir / "coverage.txt").write_text(
        f"idx={idx}\nn_mirror_px={n_mirror}\nn_rcs_covered_px={n_covered}\ncoverage={coverage:.6f}\n"
    )

    logger.info("[%d] coverage=%.4f  (%d / %d px)", idx, coverage, n_covered, n_mirror)
    return {"idx": idx, "n_mirror_px": n_mirror, "n_rcs_covered_px": n_covered, "coverage": coverage}


def main() -> None:
    parser = argparse.ArgumentParser(description="RCS mirror coverage fraction.")
    parser.add_argument("--eval-dir", type=Path, default=Path("outputs/rcs_mirrorbench_eval"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    args = parser.parse_args()

    eval_dir   = args.eval_dir
    output_dir = args.output_dir or eval_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.indices is not None:
        indices = args.indices
    else:
        indices = sorted(
            int(p.name.split("_")[1])
            for p in eval_dir.glob("sample_*")
            if p.is_dir() and p.name.split("_")[1].isdigit()
        )
        logger.info("Found %d sample dirs.", len(indices))

    print("Loading MirrorBench V2 dataset...")
    loader = MirrorBenchV2SampleLoader()
    rows: list[dict] = []

    for idx in indices:
        result = process_sample(idx, eval_dir / f"sample_{idx}", loader)
        if result is not None:
            rows.append(result)

    if not rows:
        logger.error("No results produced.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    df.to_csv(output_dir / "coverage.csv", index=False)

    mean_cov = df["coverage"].mean()
    std_cov  = df["coverage"].std()

    summary = "\n".join([
        "=" * 50,
        "RCS mask coverage of mirror region",
        f"  Samples : {len(df)}",
        f"  Mean    : {mean_cov:.4f}  (std={std_cov:.4f})",
        "=" * 50,
    ])
    print("\n" + summary)
    (output_dir / "coverage_summary.txt").write_text(summary + "\n")
    logger.info("Saved coverage.csv and coverage_summary.txt to %s", output_dir)


if __name__ == "__main__":
    main()
