from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
    point: np.ndarray
    normal: np.ndarray


def fit_plane_svd(points: np.ndarray) -> "Plane":
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane.")

    plane_point = points.mean(axis=0)
    centered = points - plane_point
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    return Plane(point=plane_point.astype(np.float32), normal=normal.astype(np.float32))


def orient_plane_toward_camera(plane: "Plane", camera_center: np.ndarray | None = None) -> "Plane":
    if camera_center is None:
        camera_center = np.zeros(3, dtype=np.float32)
    if np.dot(camera_center - plane.point, plane.normal) < 0:
        return Plane(point=plane.point, normal=-plane.normal)
    return plane
