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

from fill_my_mirror.loaders import Sample, RealImageSample, BlenderSample
from fill_my_mirror.plane import Plane, fit_plane_svd, orient_plane_toward_camera


TEMP_OUTPUT_DIR = Path("temp_outputs")
TEMP_OUTPUT_DIR.mkdir(exist_ok=True)


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

    def get_geometry(self, sample: RealImageSample) -> GeometryOutputBase:
        assert isinstance(sample, RealImageSample), (
            f"MoGeGeometryProcessor expects a RealImageSample, got {type(sample).__name__}"
        )

        if sample.mask_paths:
            return self._get_geometry_multiple_mirrors(sample)
        return self._get_geometry_single_mirror(sample)

    def _get_geometry_single_mirror(self, sample: RealImageSample) -> GeometryOutputSingleMirror:
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
        plane = orient_plane_toward_camera(fit_plane_svd(mirror_pts))
        mesh_path = _build_mesh(image, points, depth, mirror_mask)

        entry: MirrorEntry = (mirror_pts, (0,), (0, 0, 0), mesh_path, plane)
        return GeometryOutputSingleMirror(
            intrinsics=intrinsics,
            mirror_entry=entry,
        )

    def _get_geometry_multiple_mirrors(self, sample: RealImageSample) -> GeometryOutputMultipleMirrors:
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

        colored_image_path = TEMP_OUTPUT_DIR / "colored_input_image.png"
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
                output_path=TEMP_OUTPUT_DIR / f"geometry_mesh_mirror_{i}.glb",
            )
            entry: MirrorEntry = (mirror_pts_i, (i,), color_i, mesh_i_path, plane_i)
            entries.append(entry)
            
        return GeometryOutputMultipleMirrors(
            intrinsics=intrinsics,
            mirror_entries=entries,
        )


class BlenderGeometryProcessor(GeometryProcessorBase):

    def __init__(self):
        pass

    def get_geometry(self, sample: Sample) -> GeometryOutputSingleMirror:
        assert isinstance(sample, BlenderSample), (
            f"BlenderGeometryProcessor expects a BlenderSample, got {type(sample).__name__}"
        )

        image = cv2.imread(sample.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mirror_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        mirror_mask = mirror_mask > 127

        mirror_pts = sample.points[mirror_mask]
        plane = orient_plane_toward_camera(fit_plane_svd(mirror_pts))
        mesh_path = _build_mesh(image, sample.points, sample.depth, mirror_mask)

        entry: MirrorEntry = (mirror_pts, (0,), (0, 0, 0), mesh_path, plane)
        return GeometryOutputSingleMirror(
            intrinsics=sample.intrinsics,
            mirror_entry=entry,
        )


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

    mesh_mask = surface_mask & (~mirror_mask)

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


def estimate_geometry(sample: Sample, model_name: str) -> GeometryOutputBase:
    if isinstance(sample, BlenderSample):
        return BlenderGeometryProcessor().get_geometry(sample)
    return MoGeGeometryProcessor(model_name).get_geometry(sample)
