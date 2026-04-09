from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
import cv2
import trimesh
from PIL import Image

import utils3d
from moge.model.v2 import MoGeModel

from fill_my_mirror.loaders import Sample, RealImageSample, BlenderSample


TEMP_OUTPUT_DIR = Path("temp_outputs")
TEMP_OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class GeometryOutput:
    mesh_path: Path
    points: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    mirror_points: np.ndarray


class GeometryProcessorBase(ABC):

    @abstractmethod
    def get_geometry(self, sample: Sample) -> GeometryOutput: ...


class MoGeGeometryProcessor(GeometryProcessorBase):

    def __init__(self, model_name: str):
        if not torch.cuda.is_available():
            print("CUDA is not available. Inference using CPU")
            self.device = "cpu"
        else:
            self.device = "cuda"
        self.model = MoGeModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_geometry(self, sample: Sample) -> GeometryOutput:
        assert isinstance(sample, RealImageSample), (
            f"MoGeGeometryProcessor expects a RealImageSample, got {type(sample).__name__}"
        )

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
            output = self.model.infer(
                image_tensor,
                resolution_level=9,
                apply_mask=True
            )

        points = output["points"].cpu().numpy().astype(np.float32)
        points = points * np.array([-1.0, -1.0, 1.0], dtype=np.float32)
        depth = output["depth"].cpu().numpy()
        intrinsics = output["intrinsics"].cpu().numpy()

        mirror_points = points[mirror_mask]

        return GeometryOutput(
            mesh_path=_build_mesh(image, points, depth, mirror_mask),
            points=points,
            depth=depth,
            intrinsics=intrinsics,
            mirror_points=mirror_points,
        )


class BlenderGeometryProcessor(GeometryProcessorBase):

    def __init__(self):
        pass  # no model needed

    def get_geometry(self, sample: Sample) -> GeometryOutput:
        assert isinstance(sample, BlenderSample), (
            f"BlenderGeometryProcessor expects a BlenderSample, got {type(sample).__name__}"
        )

        image = cv2.imread(sample.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mirror_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        mirror_mask = mirror_mask > 127

        mirror_points = sample.points[mirror_mask]

        return GeometryOutput(
            mesh_path=_build_mesh(image, sample.points, sample.depth, mirror_mask),
            points=sample.points,
            depth=sample.depth,
            intrinsics=sample.intrinsics,
            mirror_points=mirror_points,
        )


def _build_mesh(
    image: np.ndarray,
    points: np.ndarray,
    depth: np.ndarray,
    mirror_mask: np.ndarray,
) -> Path:
    """Build and export a textured GLB mesh. Returns the path to the exported file."""

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

    mesh_path = TEMP_OUTPUT_DIR / "geometry_mesh.glb"
    mesh.export(mesh_path)
    return mesh_path


def estimate_geometry(sample: Sample, model_name: str) -> GeometryOutput:
    if isinstance(sample, BlenderSample):
        return BlenderGeometryProcessor().get_geometry(sample)
    return MoGeGeometryProcessor(model_name).get_geometry(sample)
