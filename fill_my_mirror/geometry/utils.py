from __future__ import annotations

import numpy as np


def unproject_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Unproject a depth map to (H, W, 3) camera-space points.

    Follows the normalized-intrinsics convention (fx/W, cx/W, cy/H) and
    applies the sign flip [-1, -1, 1] used throughout the pipeline.
    """
    H, W = depth.shape
    fx = intrinsics[0, 0] * W
    fy = intrinsics[1, 1] * H
    cx = intrinsics[0, 2] * W
    cy = intrinsics[1, 2] * H
    ys, xs = np.mgrid[0:H, 0:W]
    pts = np.stack(
        [(xs - cx) / (fx + 1e-8) * depth,
         (ys - cy) / (fy + 1e-8) * depth,
         depth],
        axis=-1,
    ).astype(np.float32)
    pts *= np.array([-1.0, -1.0, 1.0], dtype=np.float32)
    return pts


def align_depth_ls(d_est: np.ndarray, d_gt: np.ndarray) -> tuple[float, float]:
    """Least-squares (scale, shift) so that scale·d_est + shift ≈ d_gt over valid pixels."""
    valid = np.isfinite(d_est) & np.isfinite(d_gt) & (d_gt > 0) & (d_est > 0)
    if valid.sum() < 10:
        med_gt = float(np.median(d_gt[np.isfinite(d_gt) & (d_gt > 0)]))
        med_est = float(np.median(d_est[np.isfinite(d_est) & (d_est > 0)]))
        return med_gt / (med_est + 1e-8), 0.0
    x, y = d_est[valid].ravel(), d_gt[valid].ravel()
    A = np.stack([x, np.ones_like(x)], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])
