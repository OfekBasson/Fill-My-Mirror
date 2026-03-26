from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh

from fill_my_mirror.geometry import GeometryOutput


TEMP_OUTPUT_DIR = Path("temp_outputs")
TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Plane:
    point: np.ndarray
    normal: np.ndarray


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_binary_mask(mask_path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return mask > 127


def fit_plane_svd(points: np.ndarray) -> Plane:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane.")

    plane_point = points.mean(axis=0)
    centered = points - plane_point
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    plane_normal = vh[-1]
    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-8)

    return Plane(
        point=plane_point.astype(np.float32),
        normal=plane_normal.astype(np.float32)
    )


def orient_plane_toward_camera(plane: Plane, camera_center: np.ndarray | None = None) -> Plane:
    if camera_center is None:
        camera_center = np.zeros(3, dtype=np.float32)

    signed_distance = np.dot(camera_center - plane.point, plane.normal)
    if signed_distance < 0:
        return Plane(point=plane.point, normal=-plane.normal)
    return plane


def estimate_mirror_plane(
    geometry_output: GeometryOutput,
) -> Plane:
    masked_points = geometry_output.mirror_points
    if masked_points.shape[0] < 3:
        raise ValueError("Not enough valid masked points to estimate mirror plane.")

    plane = fit_plane_svd(masked_points)
    plane = orient_plane_toward_camera(plane)
    return plane


def reflect_points_across_plane(points: np.ndarray, plane: Plane) -> np.ndarray:
    signed_distances = np.sum((points - plane.point) * plane.normal, axis=-1, keepdims=True)
    reflected_points = points - 2.0 * signed_distances * plane.normal
    return reflected_points


def build_reflected_mesh(
    mesh_path: str | Path,
    plane: Plane,
    output_path: str | Path = TEMP_OUTPUT_DIR / "reflected_scene.glb",
) -> Path:
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a Trimesh at {mesh_path}")

    vertices = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.faces).copy()

    # Keep only geometry on the camera side of the mirror plane.
    signed_distances = np.sum((vertices - plane.point) * plane.normal, axis=1)
    keep_vertex = signed_distances >= 0.0
    keep_face = np.all(keep_vertex[faces], axis=1)

    kept_faces = faces[keep_face]
    if kept_faces.shape[0] == 0:
        raise ValueError("No faces remain after filtering the mesh by the mirror plane.")

    reflected_vertices = reflect_points_across_plane(vertices, plane)

    reflected_mesh = trimesh.Trimesh(
        vertices=reflected_vertices,
        faces=kept_faces,
        visual=mesh.visual,
        process=False,
    )
    reflected_mesh.remove_unreferenced_vertices()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reflected_mesh.export(output_path)
    return output_path


def composite_projection_onto_image(
    original_image: np.ndarray,
    rendered_image: np.ndarray,
    mirror_mask: np.ndarray,
) -> np.ndarray:
    composited = original_image.copy()
    composited[mirror_mask] = rendered_image[mirror_mask]
    return composited


def build_inpainting_mask(
    bw_rendered_image: np.ndarray,
    mirror_mask: np.ndarray,
) -> np.ndarray:
    if bw_rendered_image.ndim == 3:
        bw_gray = cv2.cvtColor(bw_rendered_image, cv2.COLOR_RGB2GRAY)
    else:
        bw_gray = bw_rendered_image

    # White means "needs inpainting":
    # - background that was not projected
    # - back-facing triangles rendered white in the BW pass
    white_region = bw_gray > 127
    inpainting_region = mirror_mask & white_region

    output = np.zeros_like(bw_gray, dtype=np.uint8)
    output[inpainting_region] = 255
    
    kernel = np.ones((5, 5), np.uint8)
    output = cv2.dilate(output, kernel, iterations=2)
    
    return output