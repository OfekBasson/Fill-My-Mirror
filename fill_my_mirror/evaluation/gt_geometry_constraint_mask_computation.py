"""
Ground-truth geometry constraint mask computation for Blender dataset samples.

Uses the existing projection pipeline with ground-truth geometry (available only
for Blender scenes) to determine which mirror pixels are geometrically constrained
by the visible scene. This is the Blender-specific counterpart to
``rcs_mask_computation``, which uses MASt3R correspondences for real images.

Requires Blender to be installed (see ``scripts/install_blender.sh``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from fill_my_mirror.loaders import BlenderSample

logger = logging.getLogger(__name__)


def compute_gt_geometry_constraint_mask(
    sample: BlenderSample,
    blender_path: str | Path,
    mask_stem: str,
) -> Path:
    """
    Compute the geometry-constrained mirror pixels mask using ground-truth geometry.

    Runs the existing projection pipeline (geometry estimation → Blender rendering)
    on a ``BlenderSample`` whose geometry fields (``points``, ``depth``,
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
    sample : BlenderSample
        A sample loaded from the Blender HuggingFace dataset.  Must have
        ``points``, ``depth``, and ``intrinsics`` populated.
    blender_path : str or Path
        Path to the Blender executable.
    mask_stem : str
        Used as the filename for the saved mask (e.g. ``"0"``).

    Returns
    -------
    Path
        Path to the saved binary mask PNG
        (``gt_geometry_constraint_masks/blender/{sample_id}.png``).
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

    # Constrained pixels: inside the mirror AND already covered by the projection
    # (i.e. NOT in the inpainting region).
    constrained = mirror_mask & ~inpainting_mask

    save_path = Path("gt_geometry_constraint_masks") / "blender" / f"{mask_stem}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((constrained.astype(np.uint8) * 255), mode="L").save(save_path)
    logger.info("Saved GT geometry constraint mask to %s", save_path)
    return save_path
