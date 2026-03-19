from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
import cv2
import trimesh
from PIL import Image

import utils3d
from moge.model.v2 import MoGeModel


TEMP_OUTPUT_DIR = Path("temp_outputs")
TEMP_OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class GeometryOutput:
    mesh_path: Path
    points: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    intrinsics: np.ndarray


class GeometryEstimator:

    def __init__(self, model_name: str):
        if not torch.cuda.is_available():
            print("CUDA is not available. Inference using CPU")
            self.device = "cpu"
        else:
            self.device = "cuda"
        self.model = MoGeModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict(self, image_path: str) -> GeometryOutput:

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        image_tensor = (
            torch.tensor(image / 255.0, dtype=torch.float32)
            .permute(2, 0, 1)
            .to(self.device)
        )

        with torch.inference_mode():
            output = self.model.infer(
                image_tensor,
                resolution_level=9,
                apply_mask=True
            )

        points = output["points"].cpu().numpy()
        depth = output["depth"].cpu().numpy()
        mask = output["mask"].cpu().numpy()
        intrinsics = output["intrinsics"].cpu().numpy()

        # ---------- compute geometry attributes ----------

        normals, normals_mask = utils3d.numpy.point_map_to_normal_map(
            points,
            mask=mask
        )

        final_mask = mask & ~(
            utils3d.numpy.depth_map_edge(depth, rtol=0.03, mask=mask) &
            utils3d.numpy.normal_map_edge(normals, tol=5, mask=normals_mask)
        )

        uv_map = utils3d.np.uv_map((height, width))
        faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
            points,
            image.astype(np.float32) / 255.0,
            uv_map,
            mask=final_mask,
            tri=True,
        )

        vertices = vertices * [1, -1, -1]
        vertex_uvs = vertex_uvs * [1, -1] + [0, 1]

        # ---------- build mesh ----------

        mesh = trimesh.Trimesh(
            vertices=vertices * [-1, 1, -1],
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

        return GeometryOutput(
            mesh_path=mesh_path,
            points=points,
            depth=depth,
            mask=mask,
            intrinsics=intrinsics
        )


def estimate_geometry(image_path: str, model_name: str):

    estimator = GeometryEstimator(
        model_name=model_name
    )

    return estimator.predict(image_path)