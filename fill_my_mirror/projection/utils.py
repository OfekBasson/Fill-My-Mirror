from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import trimesh

from fill_my_mirror.plane import Plane, fit_plane_svd, orient_plane_toward_camera
from fill_my_mirror.geometry import (
    GeometryOutputBase,
    GeometryOutputSingleMirror,
    GeometryOutputMultipleMirrors,
)


TEMP_OUTPUT_DIR = Path("temp_outputs")
TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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



def estimate_mirror_plane(
    geometry_output: GeometryOutputBase,
) -> Plane | list[Plane]:
    if isinstance(geometry_output, GeometryOutputSingleMirror):
        pts, _, _, _, _ = geometry_output.mirror_entry
        plane = fit_plane_svd(pts)
        return orient_plane_toward_camera(plane)
    elif isinstance(geometry_output, GeometryOutputMultipleMirrors):
        return [
            orient_plane_toward_camera(fit_plane_svd(pts))
            for (pts, _, _, _, _) in geometry_output.mirror_entries
        ]
    raise TypeError(f"Unknown geometry output type: {type(geometry_output)}")


def reflect_points_across_plane(points: np.ndarray, plane: Plane) -> np.ndarray:
    signed_distances = np.sum((points - plane.point) * plane.normal, axis=-1, keepdims=True)
    reflected_points = points - 2.0 * signed_distances * plane.normal
    return reflected_points


def reflect_plane_across_plane(plane: Plane, mirror_plane: Plane) -> Plane:
    new_point = reflect_points_across_plane(plane.point[None], mirror_plane)[0]
    n = mirror_plane.normal
    new_normal = plane.normal - 2 * np.dot(plane.normal, n) * n
    new_normal = new_normal / (np.linalg.norm(new_normal) + 1e-8)
    return Plane(point=new_point.astype(np.float32), normal=new_normal.astype(np.float32))


def _compact_mesh(vertices: np.ndarray, faces: np.ndarray, source_visual) -> trimesh.Trimesh:
    """Build a Trimesh referencing only the vertices used by faces, preserving UV texture."""
    used = np.unique(faces)
    remap = np.empty(len(vertices), dtype=np.intp)
    remap[used] = np.arange(len(used), dtype=np.intp)
    new_vertices = vertices[used]
    new_faces = remap[faces]

    if isinstance(source_visual, trimesh.visual.TextureVisuals) and source_visual.uv is not None:
        new_visual = trimesh.visual.TextureVisuals(
            uv=source_visual.uv[used],
            material=source_visual.material,
        )
    else:
        new_visual = source_visual

    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces, visual=new_visual, process=False)


def build_reflected_mesh(
    mesh_path: str | Path,
    plane: Plane,
    output_path: str | Path = TEMP_OUTPUT_DIR / "reflected_scene.glb",
    combined: bool = False,
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
    reflected_mesh = _compact_mesh(reflected_vertices, kept_faces, mesh.visual)

    if combined:
        original_kept = _compact_mesh(vertices, kept_faces, mesh.visual)
        combined_mesh = trimesh.util.concatenate([original_kept, reflected_mesh])
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_mesh.export(output_path)
        return output_path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reflected_mesh.export(output_path)
    print(f'Saved {"reflected" if not combined else "combined"} mesh to {output_path}')
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

    white_region = bw_gray > 127
    inpainting_region = mirror_mask & white_region

    output = np.zeros_like(bw_gray, dtype=np.uint8)
    output[inpainting_region] = 255

    kernel = np.ones((5, 5), np.uint8)
    output = cv2.dilate(output, kernel, iterations=2)

    return output


def find_color_mask(
    image: np.ndarray,
    color: tuple[int, int, int],
    region_mask: np.ndarray,
    tolerance: int = 10,
) -> np.ndarray:
    diff = np.abs(image.astype(np.int32) - np.array(color, dtype=np.int32))
    close = np.all(diff <= tolerance, axis=2)
    return close & region_mask



def depth_and_intrinsics_to_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Convert a Blender Z-depth map to a (H, W, 3) point map in MoGe camera space."""
    H, W = image_shape
    ys, xs = np.mgrid[0:H, 0:W]
    fx_n = intrinsics[0, 0]
    fy_n = intrinsics[1, 1]
    # MoGe flips x and y relative to standard camera convention
    X = -(xs / W - 0.5) / fx_n * depth
    Y = -(ys / H - 0.5) / fy_n * depth
    return np.stack([X, Y, depth], axis=-1).astype(np.float32)


def load_exr_depth(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Could not read depth EXR: {path}")
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32)
