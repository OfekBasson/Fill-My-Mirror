from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from fill_my_mirror.blender import render_with_blender
from fill_my_mirror.geometry import (
    GeometryOutputSingleMirror,
    GeometryOutputMultipleMirrors,
)
from .utils import (
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
    geometry_constraint_mask_path: Path


@dataclass
class ProjectionOutputMultipleMirrors:
    projected_image_path: Path
    geometry_constraint_masks_paths: list[Path]


def run_projection_single_mirror(
    geometry_output: GeometryOutputSingleMirror,
    image_path: str | Path,
    mirror_mask_path: str | Path,
    blender_path: str | Path,
    projected_image_path: str | Path | None = None,
    geometry_constraint_mask_path: str | Path | None = None,
    tmp_dir: str | Path | None = None,
) -> ProjectionOutput:
    tmp_dir = Path(tmp_dir) if tmp_dir is not None else TEMP_OUTPUT_DIR
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if projected_image_path is None:
        projected_image_path = tmp_dir / "projected_image.png"
    if geometry_constraint_mask_path is None:
        geometry_constraint_mask_path = tmp_dir / "geometry_constraint_mask.png"

    image = load_rgb_image(image_path)
    mirror_mask = load_binary_mask(mirror_mask_path)

    if mirror_mask.shape != image.shape[:2]:
        raise ValueError(
            f"Mirror mask shape {mirror_mask.shape} does not match image shape {image.shape[:2]}"
        )

    plane = estimate_mirror_plane(geometry_output=geometry_output)

    _, _, _, entry_mesh_path, _ = geometry_output.mirror_entry
    reflected_mesh_path = build_reflected_mesh(
        mesh_path=entry_mesh_path,
        plane=plane,
        output_path=tmp_dir / "reflected_scene.glb",
    )
    raw_render_path = tmp_dir / "reflected_scene_raw.png"
    raw_bw_render_path = tmp_dir / "reflected_bw_scene_raw.png"

    render_with_blender(
        blender_path=blender_path,
        glb_path=reflected_mesh_path,
        intrinsics=geometry_output.intrinsics,
        image_shape=image.shape[:2],
        output_path=raw_render_path,
        bw_output_path=raw_bw_render_path,
        tmp_dir=tmp_dir,
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
    geometry_constraint_mask_path = Path(geometry_constraint_mask_path)

    projected_image_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_constraint_mask_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(projected_image_path),
        cv2.cvtColor(composited, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(str(geometry_constraint_mask_path), geometry_constraint_mask)

    return ProjectionOutput(
        projected_image_path=projected_image_path,
        geometry_constraint_mask_path=geometry_constraint_mask_path,
    )



def run_projection_multiple_mirrors(
    geometry_output: GeometryOutputMultipleMirrors,
    image_path: str | Path,
    mirror_mask_paths: list[str | Path],
    blender_path: str | Path,
    projected_image_path: str | Path | None = None,
    geometry_constraint_masks_paths: list[str | Path] | None = None,
    tmp_dir: str | Path | None = None,
) -> ProjectionOutputMultipleMirrors:
    if len(mirror_mask_paths) != len(geometry_output.mirror_entries):
        raise ValueError(
            f"Expected {len(geometry_output.mirror_entries)} mirror masks, "
            f"got {len(mirror_mask_paths)}"
        )

    _tmp_dir = Path(tmp_dir) if tmp_dir is not None else TEMP_OUTPUT_DIR
    _tmp_dir.mkdir(parents=True, exist_ok=True)

    if projected_image_path is None:
        projected_image_path = _tmp_dir / "projected_image.png"

    current_image = load_rgb_image(image_path)

    if geometry_constraint_masks_paths is None:
        geometry_constraint_masks_paths = [
            _tmp_dir / f"geometry_constraint_mask_{i}.png"
            for i in range(len(geometry_output.mirror_entries))
        ]

    saved_constraint_paths: list[Path] = []

    for i, entry in enumerate(geometry_output.mirror_entries):
        _, path_tuple, _, mesh_path_i, plane_i = entry

        mirror_mask_i = load_binary_mask(mirror_mask_paths[i])

        path_str = "_".join(str(x) for x in path_tuple)

        combined_scene_mesh_i = _tmp_dir / f"combined_scene_{path_str}.glb"
        build_reflected_mesh(
            mesh_path=mesh_path_i,
            plane=plane_i,
            output_path=combined_scene_mesh_i,
            combined=True,
        )

        raw_render_i_path = _tmp_dir / f"raw_render_{path_str}.png"
        bw_render_i_path = _tmp_dir / f"bw_render_{path_str}.png"
        depth_i_path = _tmp_dir / f"depth_{path_str}.exr"

        render_with_blender(
            blender_path=blender_path,
            glb_path=combined_scene_mesh_i,
            intrinsics=geometry_output.intrinsics,
            image_shape=current_image.shape[:2],
            output_path=raw_render_i_path,
            bw_output_path=bw_render_i_path,
            depth_output_path=depth_i_path,
            tmp_dir=_tmp_dir,
        )
        raw_render_i = load_rgb_image(raw_render_i_path)
        bw_render_i = load_rgb_image(bw_render_i_path)

        composited_i = composite_projection_onto_image(current_image, raw_render_i, mirror_mask_i)

        constraint_mask_i = build_inpainting_mask(bw_render_i, mirror_mask_i)
        constraint_path_i = Path(geometry_constraint_masks_paths[i])
        constraint_path_i.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(constraint_path_i), constraint_mask_i)
        saved_constraint_paths.append(constraint_path_i)

        current_image[mirror_mask_i] = composited_i[mirror_mask_i]

    projected_image_path = Path(projected_image_path)
    projected_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(projected_image_path),
        cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR),
    )

    return ProjectionOutputMultipleMirrors(
        projected_image_path=projected_image_path,
        geometry_constraint_masks_paths=saved_constraint_paths,
    )
