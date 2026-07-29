"""
RCS Mask Evaluation on a Dataset with Known GT Masks
=====================================================
For each image in a dataset stored on R2, computes the RCS mask using a
configurable set of transforms (hflip, rot180 — union), then evaluates
precision/recall against the ground-truth constrained pixels mask.

Expected R2 layout under <r2-prefix>/<idx>/:
  gt_image.png
  generative_refinement_mask.png
  constrained_pixels_gt_geometry_mask.png

Usage
-----
    # MirrorBench (default):
    conda run -n fill-my-mirror python scripts/evaluate_rcs_on_dataset_with_known_gt_masks.py \
        --r2-prefix mirrorbench_v2/gt_geometry \
        --transforms hflip rot180 \
        --output-dir outputs/rcs_mirrorbench_eval

    # Blender:
    conda run -n fill-my-mirror python scripts/evaluate_rcs_on_dataset_with_known_gt_masks.py \
        --r2-prefix blender/gt_geometry \
        --transforms hflip \
        --output-dir outputs/rcs_blender_eval

    # Run all indices:
    conda run -n fill-my-mirror python scripts/evaluate_rcs_on_dataset_with_known_gt_masks.py \
        --r2-prefix blender/gt_geometry \
        --output-dir outputs/rcs_blender_eval
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fill_my_mirror.evaluation.rcs_mask_computation import (
    _ensure_mast3r,
    _run_mast3r_correspondences,
    _dilate_and_intersect,
    _pil_to_uint8_rgb,
    _pil_to_binary,
    _MAST3R_IMG_SIZE,
)
from fill_my_mirror.storage import R2Client

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DILATION_RADIUS = 5
DILATION_ITERATIONS = 1
RCS_COMPUTE_SIZE = 800  # all RCS computation (dilation included) runs at this resolution

ALL_TRANSFORMS = ["hflip", "rot180"]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _build_scene(image_arr: np.ndarray, mirror_mask_arr: np.ndarray) -> np.ndarray:
    scene = image_arr.copy()
    scene[mirror_mask_arr] = 0
    return scene


def _build_mirror(image_arr: np.ndarray, mirror_mask_arr: np.ndarray, transform: str) -> np.ndarray:
    mirror = image_arr.copy()
    mirror[~mirror_mask_arr] = 0
    if transform == "hflip":
        return np.fliplr(mirror)
    elif transform == "rot180":
        return np.rot90(mirror, 2)
    else:
        raise ValueError(f"Unknown transform: {transform!r}")


def _undo_base_transform(pts: np.ndarray, transform: str, S: int = _MAST3R_IMG_SIZE) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    if transform == "hflip":
        x = S - 1 - x
    elif transform == "rot180":
        x = S - 1 - x
        y = S - 1 - y
    return np.stack([x, y], axis=1)


def _mirror_pts_to_image(
    pts_mirror_raw: np.ndarray,
    transform: str,
    W: int,
    H: int,
    S: int = _MAST3R_IMG_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    pts = _undo_base_transform(pts_mirror_raw, transform, S)
    xs = np.clip(np.round(pts[:, 0] * (W / S)).astype(int), 0, W - 1)
    ys = np.clip(np.round(pts[:, 1] * (H / S)).astype(int), 0, H - 1)
    return xs, ys


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_pr(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# R2 helpers
# ---------------------------------------------------------------------------

def _download_pil(r2: R2Client, key: str) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        r2.download_file(key, Path(tf.name))
        return Image.open(tf.name).copy()


# ---------------------------------------------------------------------------
# Per-sample RCS computation
# ---------------------------------------------------------------------------

def compute_rcs_for_sample(
    idx: int,
    r2: R2Client,
    mast3r_model_name: str,
    device: str,
    output_dir: Path,
    r2_prefix: str,
    transforms: list[str],
) -> dict | None:
    prefix = f"{r2_prefix}/{idx}"
    image_key  = f"{prefix}/gt_image.png"
    mask_key   = f"{prefix}/generative_refinement_mask.png"
    gt_key     = f"{prefix}/constrained_pixels_gt_geometry_mask.png"

    try:
        image_pil  = _download_pil(r2, image_key)
        mask_pil   = _download_pil(r2, mask_key)
        gt_pil     = _download_pil(r2, gt_key)
    except Exception as e:
        logger.warning("[%d] Failed to download R2 assets: %s", idx, e)
        return None

    image_arr   = _pil_to_uint8_rgb(image_pil)
    mirror_mask = _pil_to_binary(mask_pil)
    gt_mask     = np.asarray(gt_pil.convert("L"), dtype=np.uint8) > 127
    H, W        = image_arr.shape[:2]

    # Resize to RCS_COMPUTE_SIZE for consistent dilation scale
    S = RCS_COMPUTE_SIZE
    image_800 = np.array(Image.fromarray(image_arr).resize((S, S), Image.LANCZOS))
    mirror_mask_800 = np.array(
        Image.fromarray(mirror_mask.astype(np.uint8) * 255).resize((S, S), Image.NEAREST)
    ) > 127

    scene_arr = _build_scene(image_800, mirror_mask_800)

    # Union of correspondences across requested transforms (computed at RCS_COMPUTE_SIZE)
    combined_mask_800 = np.zeros((S, S), dtype=bool)
    for transform in transforms:
        mirror_arr = _build_mirror(image_800, mirror_mask_800, transform)
        pts_scene_raw, pts_mirror_raw = _run_mast3r_correspondences(
            scene_arr, mirror_arr, mast3r_model_name, device,
        )
        n_pts = pts_mirror_raw.shape[0]
        logger.info("[%d] %s  correspondences: %d", idx, transform, n_pts)
        if n_pts > 0:
            xs, ys = _mirror_pts_to_image(pts_mirror_raw, transform, S, S)
            combined_mask_800[ys, xs] = True

    rcs_mask_800 = _dilate_and_intersect(combined_mask_800, mirror_mask_800, DILATION_RADIUS, iterations=DILATION_ITERATIONS)

    # Resize results back to original resolution
    combined_mask = np.array(
        Image.fromarray(combined_mask_800.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
    ) > 127
    rcs_mask = np.array(
        Image.fromarray(rcs_mask_800.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
    ) > 127

    precision, recall, f1 = _compute_pr(rcs_mask, gt_mask)
    logger.info("[%d] P=%.3f  R=%.3f  F1=%.3f", idx, precision, recall, f1)

    # Save outputs
    sample_dir = output_dir / f"sample_{idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((combined_mask.astype(np.uint8) * 255), mode="L").save(
        sample_dir / "combined_correspondence_mask.png"
    )
    Image.fromarray((rcs_mask.astype(np.uint8) * 255), mode="L").save(
        sample_dir / "rcs_mask.png"
    )
    Image.fromarray((gt_mask.astype(np.uint8) * 255), mode="L").save(
        sample_dir / "constrained_pixels_gt_geometry_mask.png"
    )
    Image.fromarray((mirror_mask.astype(np.uint8) * 255), mode="L").save(
        sample_dir / "generative_refinement_mask.png"
    )

    return {
        "idx":       idx,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "n_rcs_px":  int(rcs_mask.sum()),
        "n_gt_px":   int(gt_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Evaluate RCS mask (r={DILATION_RADIUS}, i={DILATION_ITERATIONS}) on a dataset with known GT masks."
    )
    parser.add_argument(
        "--r2-prefix", type=str, required=True,
        help="R2 key prefix for the dataset, e.g. 'mirrorbench_v2/gt_geometry' or 'blender/gt_geometry'.",
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", choices=ALL_TRANSFORMS, default=["hflip"],
        help="Transforms to use for RCS computation (union). Choices: hflip, rot180. Default: hflip.",
    )
    parser.add_argument(
        "--indices", type=int, nargs="*", default=None,
        help="Subset of dataset indices. Default: all available.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for results.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device (cuda/cpu).")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip indices where rcs_mask.png already exists in the output dir.",
    )
    args = parser.parse_args()

    r2_prefix = args.r2_prefix
    transforms = args.transforms
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset prefix : %s", r2_prefix)
    logger.info("Transforms     : %s", transforms)

    if not _ensure_mast3r():
        logger.error("MASt3R not available. Aborting.")
        sys.exit(1)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

    r2 = R2Client()

    if args.indices is not None:
        indices = args.indices
    else:
        logger.info("Listing available indices from R2 under %s ...", r2_prefix)
        keys = r2.list_keys(r2_prefix + "/")
        seen: set[int] = set()
        for k in keys:
            parts = k.removeprefix(r2_prefix + "/").split("/")
            if parts[0].isdigit():
                seen.add(int(parts[0]))
        indices = sorted(seen)
        logger.info("Found %d indices.", len(indices))

    rows: list[dict] = []
    metrics_path = output_dir / "metrics.csv"

    if metrics_path.exists():
        existing_df = pd.read_csv(metrics_path)
        existing_indices = set(existing_df["idx"].tolist())
        rows = existing_df.to_dict("records")
    else:
        existing_indices = set()

    for idx in indices:
        if args.skip_existing and idx in existing_indices:
            logger.info("[%d] Skipping (already in metrics.csv).", idx)
            continue
        if args.skip_existing and (output_dir / f"sample_{idx}" / "rcs_mask.png").exists():
            logger.info("[%d] Skipping (rcs_mask.png already exists).", idx)
            continue

        result = compute_rcs_for_sample(
            idx, r2, mast3r_model_name, device, output_dir, r2_prefix, transforms,
        )
        if result is not None:
            rows.append(result)
            pd.DataFrame(rows).to_csv(metrics_path, index=False)

    if not rows:
        logger.error("No results produced.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    df.to_csv(metrics_path, index=False)
    logger.info("Saved metrics to %s", metrics_path)

    ranking = df.sort_values("f1", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    ranking_path = output_dir / "ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    logger.info("Saved ranking to %s", ranking_path)

    mean_p  = df["precision"].mean()
    mean_r  = df["recall"].mean()
    mean_f1 = df["f1"].mean()

    transforms_str = "+".join(transforms)
    print("\n" + "=" * 60)
    print(f"RCS Evaluation  [{r2_prefix}]  transforms={transforms_str}  (r={DILATION_RADIUS}, i={DILATION_ITERATIONS})")
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
    ax.set_title(f"RCS  [{r2_prefix}]  {transforms_str}  (r={DILATION_RADIUS}, i={DILATION_ITERATIONS})", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_scatter.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scatter plot to %s", output_dir / "precision_recall_scatter.pdf")


if __name__ == "__main__":
    main()
