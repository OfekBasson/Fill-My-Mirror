from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fill_my_mirror.blender_render import render_with_blender
from fill_my_mirror.geometry import GeometryOutput
from fill_my_mirror.projection_utils import (
    TEMP_OUTPUT_DIR,
    build_inpainting_mask,
    build_reflected_mesh,
    composite_projection_onto_image,
    estimate_mirror_plane,
    load_binary_mask,
    load_rgb_image,
)


@dataclass
class ProjectionOutput:
    projected_image_path: Path
    inpainting_mask_path: Path
    reflected_mesh_path: Path
    plane_point: np.ndarray
    plane_normal: np.ndarray


def run_projection(
    geometry_output: GeometryOutput,
    image_path: str | Path,
    mirror_mask_path: str | Path,
    blender_path: str | Path,
    projected_image_path: str | Path = TEMP_OUTPUT_DIR / "projected_image.png",
    inpainting_mask_path: str | Path = TEMP_OUTPUT_DIR / "inpainting_mask.png",
) -> ProjectionOutput:
    image = load_rgb_image(image_path)
    mirror_mask = load_binary_mask(mirror_mask_path)

    if mirror_mask.shape != image.shape[:2]:
        raise ValueError(
            f"Mirror mask shape {mirror_mask.shape} does not match image shape {image.shape[:2]}"
        )

    plane = estimate_mirror_plane(
        geometry_output=geometry_output,
    )

    reflected_mesh_path = build_reflected_mesh(
        mesh_path=geometry_output.mesh_path,
        plane=plane,
    )
    raw_render_path = TEMP_OUTPUT_DIR / "raw_projection.png"
    raw_bw_render_path = TEMP_OUTPUT_DIR / "raw_projection_bw.png"

    render_with_blender(
        blender_path=blender_path,
        glb_path=reflected_mesh_path,
        intrinsics=geometry_output.intrinsics,
        image_shape=image.shape[:2],
        output_path=raw_render_path,
        bw_output_path=raw_bw_render_path,
        front_back_facing_flip=False,
    )

    rendered_image = load_rgb_image(raw_render_path)
    bw_rendered_image = load_rgb_image(raw_bw_render_path)

    composited = composite_projection_onto_image(
        original_image=image,
        rendered_image=rendered_image,
        mirror_mask=mirror_mask,
    )

    geometry_constraint_mask = build_inpainting_mask(
        bw_rendered_image=bw_rendered_image,
        mirror_mask=mirror_mask,
    )

    projected_image_path = Path(projected_image_path)
    inpainting_mask_path = Path(inpainting_mask_path)

    projected_image_path.parent.mkdir(parents=True, exist_ok=True)
    inpainting_mask_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(projected_image_path),
        cv2.cvtColor(composited, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(str(inpainting_mask_path), geometry_constraint_mask)

    return ProjectionOutput(
        projected_image_path=projected_image_path,
        inpainting_mask_path=inpainting_mask_path,
        reflected_mesh_path=reflected_mesh_path,
        plane_point=plane.point,
        plane_normal=plane.normal,
    )