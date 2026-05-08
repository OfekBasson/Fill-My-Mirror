from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field

import torch
import numpy as np
import cv2
import trimesh
from PIL import Image

import utils3d
from moge.model.v2 import MoGeModel

from fill_my_mirror.loaders import Sample, EstimatedGeometrySample, GTGeometrySample
from fill_my_mirror.plane import Plane, fit_plane_svd, orient_plane_toward_camera


TEMP_OUTPUT_DIR = Path("temp_outputs")
TEMP_OUTPUT_DIR.mkdir(exist_ok=True)

_MIN_FINITE_RATIO = 0.01


class LowFiniteMirrorPointsRatioError(RuntimeError):
    """Raised when the ratio of finite to total 3D points inside the mirror mask is too low.

    This typically happens when the geometry model (e.g. MoGe) cannot reconstruct
    the mirror surface because it was masked out during inference. With fewer than
    1% finite points the fitted plane is unreliable and projection is skipped.
    """
    def __init__(self, finite: int, total: int):
        self.finite = finite
        self.total = total
        self.ratio = finite / total if total > 0 else 0.0
        super().__init__(
            f"Only {finite}/{total} ({self.ratio:.1%}) mirror points are finite — "
            f"below the {_MIN_FINITE_RATIO:.0%} threshold. "
            "The geometry model could not reconstruct the mirror surface. "
            "Projection will be skipped and the mirror filled entirely by inpainting."
        )


# (3D-points, path-tuple, rgb-color, per-mirror-mesh-path, plane)
MirrorEntry = tuple[np.ndarray, tuple[int, ...], tuple[int, int, int], Path, Plane]


@dataclass
class GeometryOutputBase:
    intrinsics: np.ndarray


@dataclass
class GeometryOutputSingleMirror(GeometryOutputBase):
    mirror_entry: MirrorEntry | None = None


@dataclass
class GeometryOutputMultipleMirrors(GeometryOutputBase):
    mirror_entries: list[MirrorEntry] = field(default_factory=list)

    def __post_init__(self):
        assert len(self.mirror_entries) >= 1, (
            "GeometryOutputMultipleMirrors requires at least 1 MirrorEntry"
        )


class GeometryProcessorBase(ABC):

    @abstractmethod
    def get_geometry(self, sample: Sample) -> GeometryOutputBase: ...



def _pick_unique_color(
    image: np.ndarray,
    excluded_colors: list[tuple[int, int, int]],
    n_samples: int = 5000,
) -> tuple[int, int, int]:
    vals = [0, 64, 128, 192, 255]
    candidates = [
        (r, g, b) for r in vals for g in vals for b in vals
        if max(abs(r - g), abs(g - b), abs(r - b)) > 64
        and (r, g, b) not in excluded_colors
    ]
    if not candidates:
        raise RuntimeError("Color palette exhausted — too many mirrors")

    flat = image.reshape(-1, 3).astype(np.float32)
    idx = np.random.choice(len(flat), min(n_samples, len(flat)), replace=False)
    sampled = flat[idx]

    avoid_list: list[np.ndarray] = [sampled]
    for c in excluded_colors:
        avoid_list.append(np.array(c, dtype=np.float32)[None])
    avoid = np.vstack(avoid_list)

    best: tuple[int, int, int] | None = None
    best_d = -1.0
    for c in candidates:
        ca = np.array(c, dtype=np.float32)
        d = float(np.min(np.linalg.norm(avoid - ca, axis=1)))
        if d > best_d:
            best_d, best = d, c

    assert best is not None
    return best


class MoGeGeometryProcessor(GeometryProcessorBase):

    def __init__(self, model_name: str):
        if not torch.cuda.is_available():
            print("CUDA is not available. Inference using CPU")
            self.device = "cpu"
        else:
            self.device = "cuda"
        self.model = MoGeModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_geometry(self, sample: EstimatedGeometrySample, tmp_dir: Path | None = None) -> GeometryOutputBase:
        assert isinstance(sample, EstimatedGeometrySample), (
            f"MoGeGeometryProcessor expects a RealImageSample, got {type(sample).__name__}"
        )
        tmp_dir = tmp_dir or TEMP_OUTPUT_DIR

        if sample.mask_paths:
            return self._get_geometry_multiple_mirrors(sample, tmp_dir=tmp_dir)
        return self._get_geometry_single_mirror(sample, tmp_dir=tmp_dir)

    def _get_geometry_single_mirror(self, sample: EstimatedGeometrySample, tmp_dir: Path | None = None) -> GeometryOutputSingleMirror:
        tmp_dir = tmp_dir or TEMP_OUTPUT_DIR
        image = cv2.imread(sample.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_tensor = (
            torch.tensor(image / 255.0, dtype=torch.float32)
            .permute(2, 0, 1)
            .to(self.device)
        )

        mirror_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        if mirror_mask is None:
            raise FileNotFoundError(f"Could not read mirror mask: {sample.mask_path}")
        mirror_mask = mirror_mask > 127

        if mirror_mask.shape != image.shape[:2]:
            raise ValueError(
                f"Mirror mask shape {mirror_mask.shape} does not match image shape {image.shape[:2]}"
            )

        with torch.inference_mode():
            output = self.model.infer(image_tensor, resolution_level=9, apply_mask=True)

        points = output["points"].cpu().numpy().astype(np.float32)
        points = points * np.array([-1.0, -1.0, 1.0], dtype=np.float32)
        depth = output["depth"].cpu().numpy()
        intrinsics = output["intrinsics"].cpu().numpy()

        mirror_pts = points[mirror_mask]
        finite_mirror_pts = mirror_pts[np.isfinite(mirror_pts).all(axis=1)]
        ratio = len(finite_mirror_pts) / len(mirror_pts) if len(mirror_pts) > 0 else 0.0
        if ratio < _MIN_FINITE_RATIO:
            raise LowFiniteMirrorPointsRatioError(len(finite_mirror_pts), len(mirror_pts))
        plane = orient_plane_toward_camera(fit_plane_svd(mirror_pts))
        import json
        debug_info = {
            "mirror_pts_total": int(mirror_pts.shape[0]),
            "mirror_pts_finite": int(finite_mirror_pts.shape[0]),
            "finite_ratio": round(ratio, 6),
            "mirror_pts_min": finite_mirror_pts.min(axis=0).tolist() if len(finite_mirror_pts) else None,
            "mirror_pts_max": finite_mirror_pts.max(axis=0).tolist() if len(finite_mirror_pts) else None,
            "mirror_pts_mean": finite_mirror_pts.mean(axis=0).tolist() if len(finite_mirror_pts) else None,
            "mirror_pts_std": finite_mirror_pts.std(axis=0).tolist() if len(finite_mirror_pts) else None,
            "plane_point": plane.point.tolist(),
            "plane_normal": plane.normal.tolist(),
        }
        (tmp_dir / "debug_plane.json").write_text(json.dumps(debug_info, indent=2))
        mesh_path = _build_mesh(image, points, depth, mirror_mask, output_path=tmp_dir / "geometry_mesh.glb")
        _debug_export_plane(plane, finite_mirror_pts, output_path=tmp_dir / "debug_plane.glb")

        entry: MirrorEntry = (mirror_pts, (0,), (0, 0, 0), mesh_path, plane)
        return GeometryOutputSingleMirror(
            intrinsics=intrinsics,
            mirror_entry=entry,
        )

    def _get_geometry_multiple_mirrors(self, sample: EstimatedGeometrySample, tmp_dir: Path | None = None) -> GeometryOutputMultipleMirrors:
        tmp_dir = tmp_dir or TEMP_OUTPUT_DIR
        image = cv2.imread(sample.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks: list[np.ndarray] = []
        for mask_path in sample.mask_paths:
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(f"Could not read mirror mask: {mask_path}")
            masks.append(m > 127)

        for m in masks:
            if m.shape != image.shape[:2]:
                raise ValueError(
                    f"Mirror mask shape {m.shape} does not match image shape {image.shape[:2]}"
                )

        # Assign a unique color to each mirror and paint it into a copy of the image
        colored_image = image.copy()
        excluded_colors: list[tuple[int, int, int]] = []
        colors: list[tuple[int, int, int]] = []
        for mask in masks:
            color = _pick_unique_color(colored_image, excluded_colors)
            excluded_colors.append(color)
            colors.append(color)
            colored_image[mask] = color

        colored_image_path = tmp_dir / "colored_input_image.png"
        cv2.imwrite(str(colored_image_path), cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR))

        image_tensor = (
            torch.tensor(colored_image / 255.0, dtype=torch.float32)
            .permute(2, 0, 1)
            .to(self.device)
        )

        with torch.inference_mode():
            output = self.model.infer(image_tensor, resolution_level=9, apply_mask=True)

        points = output["points"].cpu().numpy().astype(np.float32)
        points = points * np.array([-1.0, -1.0, 1.0], dtype=np.float32)
        depth = output["depth"].cpu().numpy()
        intrinsics = output["intrinsics"].cpu().numpy()

        # Build one per-mirror mesh: excludes only mirror i so mirror j≠i appears as colored geometry
        entries: list[MirrorEntry] = []
        for i, (mask_i, color_i) in enumerate(zip(masks, colors)):
            mirror_pts_i = points[mask_i]
            plane_i = orient_plane_toward_camera(fit_plane_svd(mirror_pts_i))
            mesh_i_path = _build_mesh(
                colored_image, points, depth, mask_i,
                output_path=tmp_dir / f"geometry_mesh_mirror_{i}.glb",
            )
            entry: MirrorEntry = (mirror_pts_i, (i,), color_i, mesh_i_path, plane_i)
            entries.append(entry)
            
        return GeometryOutputMultipleMirrors(
            intrinsics=intrinsics,
            mirror_entries=entries,
        )


class DepthAnythingGeometryProcessor(GeometryProcessorBase):

    def __init__(self, model_name: str):
        from depth_anything_3.api import DepthAnything3
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DepthAnything3.from_pretrained(model_name).to(self.device)

    def get_geometry(self, sample: Sample, tmp_dir: Path | None = None) -> GeometryOutputBase:
        if isinstance(sample.mask_paths, list) and len(sample.mask_paths) > 1:
            raise NotImplementedError("Multi-mirror not yet supported for DepthAnythingGeometryProcessor")
        return self._get_geometry_single_mirror(sample, tmp_dir=tmp_dir)

    def _get_geometry_single_mirror(self, sample, tmp_dir: Path | None = None) -> GeometryOutputSingleMirror:
        image_bgr = cv2.imread(str(sample.image_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask_path = sample.mask_path or (sample.mask_paths[0] if sample.mask_paths else None)
        mirror_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0

        H_orig, W_orig = image_rgb.shape[:2]
        prediction = self.model.inference(
            [str(sample.image_path)],
            process_res=max(H_orig, W_orig),
            process_res_method="upper_bound_resize",
        )

        depth = prediction.depth[0]

        H_d, W_d = depth.shape

        if prediction.intrinsics is not None:
            intrinsics_px = prediction.intrinsics[0].copy()
            # The model may process internally at a different aspect ratio than the
            # output depth resolution (e.g. crops/pads height). fx is calibrated for
            # W_d correctly; fy may reflect a different internal height. Force square
            # pixels (fy == fx) which is valid for all consumer cameras.
            fx_px = float(intrinsics_px[0, 0])
            fy_px = fx_px
            cx_px = W_d / 2.0
            cy_px = H_d / 2.0
        else:
            # Depth-only models (e.g. DA3MONO-LARGE) don't estimate intrinsics.
            # Assume a 60° horizontal FoV, principal point at image center.
            fx_px = W_d / (2 * np.tan(np.deg2rad(60) / 2))
            fy_px = fx_px
            cx_px = W_d / 2.0
            cy_px = H_d / 2.0

        H, W = depth.shape
        if mirror_mask.shape != (H, W):
            mirror_mask = cv2.resize(
                mirror_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        X = (xs - cx_px) * depth / fx_px
        Y = (ys - cy_px) * depth / fy_px
        Z = depth
        points = np.stack([X, Y, Z], axis=-1).astype(np.float32)

        points = points * np.array([-1.0, -1.0, 1.0], dtype=np.float32)

        # Normalize intrinsics to match MoGe's format (fx/W, cx/W, cy/H = 0.5)
        # so the rest of the pipeline (Blender camera setup) handles them uniformly.
        intrinsics = np.array([
            [fx_px / W_d, 0,           cx_px / W_d],
            [0,           fy_px / H_d, cy_px / H_d],
            [0,           0,           1          ],
        ], dtype=np.float32)

        mirror_pts = points[mirror_mask]
        if mirror_pts.shape[0] < 3:
            return GeometryOutputSingleMirror(intrinsics=intrinsics, mirror_entry=None)

        _tmp_dir = tmp_dir or TEMP_OUTPUT_DIR
        plane = orient_plane_toward_camera(fit_plane_svd(mirror_pts))
        mesh_path = _build_mesh(image_rgb, points, depth, mirror_mask, output_path=_tmp_dir / "geometry_mesh.glb")

        sensor_width_mm = 36.0  # Blender default
        fx_norm = float(intrinsics[0, 0])
        focal_length_mm = fx_norm * sensor_width_mm
        print("[DepthAnything intrinsics debug]")
        print(f"  depth shape:          ({H_d}, {W_d})")
        print(f"  fx_px:                {fx_px:.4f}")
        print(f"  intrinsics (normalized):\n{intrinsics}")
        print(f"  → fx_norm:            {fx_norm:.6f}")
        print(f"  → focal_length_mm:    {focal_length_mm:.4f}  (sensor_width={sensor_width_mm}mm)")

        entry: MirrorEntry = (mirror_pts, (0,), (0, 0, 0), mesh_path, plane)
        return GeometryOutputSingleMirror(intrinsics=intrinsics, mirror_entry=entry)


GEOMETRY_MODEL_REGISTRY: dict[str, type] = {
    "Ruicheng/moge-2-vitl-normal": MoGeGeometryProcessor,
    # Depth Anything 3 — requires optional install, see README step 6
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1": DepthAnythingGeometryProcessor,

}


def estimate_geometry(sample: Sample, model_name: str, tmp_dir: Path | None = None) -> GeometryOutputBase:
    tmp_dir = tmp_dir or TEMP_OUTPUT_DIR
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(sample, GTGeometrySample):
        return BlenderGeometryProcessor().get_geometry(sample, tmp_dir=tmp_dir)

    processor_class = GEOMETRY_MODEL_REGISTRY.get(model_name)
    if processor_class is None:
        if model_name.startswith("depth-anything/"):
            processor_class = DepthAnythingGeometryProcessor
        else:
            supported = ", ".join(sorted(GEOMETRY_MODEL_REGISTRY))
            raise ValueError(f"Unsupported geometry model `{model_name}`. Supported: {supported}")

    return processor_class(model_name).get_geometry(sample, tmp_dir=tmp_dir)


class BlenderGeometryProcessor(GeometryProcessorBase):

    def __init__(self):
        pass

    def get_geometry(self, sample: Sample, tmp_dir: Path | None = None) -> GeometryOutputSingleMirror:
        assert isinstance(sample, GTGeometrySample), (
            f"BlenderGeometryProcessor expects a BlenderSample, got {type(sample).__name__}"
        )
        tmp_dir = tmp_dir or TEMP_OUTPUT_DIR

        image = cv2.imread(sample.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mirror_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        mirror_mask = mirror_mask > 127

        mirror_pts = sample.points[mirror_mask]
        plane = orient_plane_toward_camera(fit_plane_svd(mirror_pts))
        mesh_path = _build_mesh(image, sample.points, sample.depth, mirror_mask, output_path=tmp_dir / "geometry_mesh.glb")

        entry: MirrorEntry = (mirror_pts, (0,), (0, 0, 0), mesh_path, plane)
        return GeometryOutputSingleMirror(
            intrinsics=sample.intrinsics,
            mirror_entry=entry,
        )


def _debug_export_plane(plane: Plane, mirror_pts: np.ndarray, output_path: Path) -> None:
    """Export a flat quad mesh visualising the fitted plane. DEBUG — remove before release."""
    normal = plane.normal / (np.linalg.norm(plane.normal) + 1e-8)
    center = plane.point

    # Build two orthogonal tangent vectors in the plane
    ref = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u) + 1e-8
    v = np.cross(normal, u)
    v /= np.linalg.norm(v) + 1e-8

    # Size quad to span the mirror point cloud
    proj_u = mirror_pts @ u
    proj_v = mirror_pts @ v
    half_u = (proj_u.max() - proj_u.min()) / 2 * 1.1
    half_v = (proj_v.max() - proj_v.min()) / 2 * 1.1

    vertices = np.array([
        center - half_u * u - half_v * v,
        center + half_u * u - half_v * v,
        center + half_u * u + half_v * v,
        center - half_u * u + half_v * v,
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    plane_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    plane_mesh.visual.vertex_colors = np.array([[255, 0, 0, 180]] * 4, dtype=np.uint8)
    plane_mesh.export(output_path)
    print(f"[debug] plane mesh saved to {output_path}")


def _build_mesh(
    image: np.ndarray,
    points: np.ndarray,
    depth: np.ndarray,
    mirror_mask: np.ndarray,
    output_path: Path | None = None,
) -> Path:
    if output_path is None:
        output_path = TEMP_OUTPUT_DIR / "geometry_mesh.glb"

    height, width = image.shape[:2]

    normals, normals_mask = utils3d.numpy.point_map_to_normal_map(
        points,
        mask=np.ones((height, width), dtype=bool),
    )

    surface_mask = ~(
        utils3d.numpy.depth_map_edge(depth, rtol=0.03) &
        utils3d.numpy.normal_map_edge(normals, tol=5, mask=normals_mask)
    )

    finite_mask = np.isfinite(points).all(axis=-1)
    mesh_mask = surface_mask & (~mirror_mask) & finite_mask

    uv_map = utils3d.np.uv_map((height, width))
    faces, vertices, _, vertex_uvs = utils3d.np.build_mesh_from_map(
        points,
        image.astype(np.float32) / 255.0,
        uv_map,
        mask=mesh_mask,
        tri=True,
    )
    vertex_uvs = vertex_uvs * [1, -1] + [0, 1]

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(image),
                metallicFactor=0.5,
                roughnessFactor=1.0
            )
        ),
        process=False
    )

    mesh.export(output_path)
    print("Mesh saved to:", output_path)
    return output_path
