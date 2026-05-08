"""
Constrained-pixels GT-geometry mask computation for Blender and MirrorBench-V2 dataset samples.

Uses the existing projection pipeline with ground-truth geometry to determine which
mirror pixels are geometrically constrained by the visible scene. This is the
GT-geometry counterpart to ``rcs_mask_computation``, which uses MASt3R correspondences
for real images.

Requires Blender to be installed (see ``scripts/install_blender.sh``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from fill_my_mirror.loaders import GTGeometrySample

logger = logging.getLogger(__name__)


def compute_constrained_pixels_gt_geometry_mask(
    sample: GTGeometrySample,
    blender_path: str | Path,
    mask_stem: str,
) -> Path:
    """
    Compute the constrained-pixels mask using ground-truth geometry.

    Runs the existing projection pipeline (geometry estimation → Blender rendering)
    on a ``GTGeometrySample`` whose geometry fields (``points``, ``depth``,
    ``intrinsics``) are already populated with ground-truth values.

    The constrained region is defined as mirror pixels whose appearance is
    **deterministically fixed** by the projected scene geometry — i.e. pixels
    that are covered by the Blender render.  Concretely:

        constrained = mirror_mask  AND  NOT(geometry_constraint_mask)

    where ``geometry_constraint_mask`` is the inpainting mask produced by the
    projection pipeline (marks pixels that still need inpainting, i.e. are *not*
    covered by the projection).

    Parameters
    ----------
    sample : GTGeometrySample
        A sample with ``points``, ``depth``, and ``intrinsics`` populated.
    blender_path : str or Path
        Path to the Blender executable.
    mask_stem : str
        Used as the filename for the saved mask (e.g. ``"0"``).

    Returns
    -------
    Path
        Path to the saved binary mask PNG
        (``constrained_pixels_gt_geometry_masks/{mask_stem}.png``).
    """
    from fill_my_mirror.geometry import estimate_geometry
    from fill_my_mirror.projection import run_projection_single_mirror

    geometry_output = estimate_geometry(sample, model_name=None)
    projection_output = run_projection_single_mirror(
        geometry_output=geometry_output,
        image_path=sample.image_path,
        mirror_mask_path=sample.mask_path,
        blender_path=blender_path,
    )

    mirror_mask = np.asarray(
        Image.open(sample.mask_path).convert("L"), dtype=np.uint8
    ) > 127
    inpainting_mask = np.asarray(
        Image.open(projection_output.geometry_constraint_mask_path).convert("L"),
        dtype=np.uint8,
    ) > 127

    constrained = mirror_mask & ~inpainting_mask

    save_path = Path("constrained_pixels_gt_geometry_masks") / f"{mask_stem}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((constrained.astype(np.uint8) * 255), mode="L").save(save_path)
    logger.info("Saved constrained-pixels GT-geometry mask to %s", save_path)
    return save_path
