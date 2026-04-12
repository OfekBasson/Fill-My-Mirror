"""
Reflection Consistency Score (RCS) mask computation using MASt3R.

Identifies mirror pixels that are geometrically constrained by the visible scene
by finding dense pixel correspondences between the scene view and a horizontally
flipped mirror view (paper Section 4.3).

Requires MASt3R to be installed:
    pip install -e third_party/MASt3R
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)

try:
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    _MAST3R_AVAILABLE = True
except ImportError:
    _MAST3R_AVAILABLE = False
    logger.warning(
        "MASt3R is not installed. RCS mask computation will be unavailable. "
        "Install with: pip install -e third_party/MASt3R"
    )

# Module-level model cache: avoids reloading on every call in a batch run.
_MODEL_CACHE: dict[tuple[str, str], "AsymmetricMASt3R"] = {}

_MAST3R_IMG_SIZE = 512


def compute_rcs_mask(
    gt_image: Image.Image,
    mirror_mask: Image.Image,
    dataset_type: str,
    mask_stem: str,
    mast3r_model_name: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    device: Optional[str] = None,
    dilation_radius: int = 5,
) -> Path:
    """
    Compute the Reflection Consistency Score mask and save it to disk.

    Decomposes the input image into a scene view and a horizontally-flipped mirror
    view, runs MASt3R to find pixel correspondences, marks mirror pixels that have a
    valid scene correspondence as constrained, dilates the result, and intersects with
    the mirror mask.

    Parameters
    ----------
    gt_image : PIL.Image
        The image with the mirror region already filled (ground-truth or inpainted).
    mirror_mask : PIL.Image
        Binary mask of the mirror region (white = mirror).
    dataset_type : str
        One of ``"blender"``, ``"real_images"``, or ``"provided_images"``.
        Determines the save subdirectory.
    mask_stem : str
        HuggingFace dataset index (as string) or image filename stem.
        Used as the saved mask filename.
    mast3r_model_name : str
        HuggingFace checkpoint identifier for MASt3R.
    device : str, optional
        Torch device (e.g. ``"cuda"`` or ``"cpu"``). Defaults to CUDA if available.
    dilation_radius : int
        Structuring-element radius for dilating the correspondence mask before
        intersection. Set to 0 to disable dilation.

    Returns
    -------
    Path
        Path to the saved binary mask PNG (white = constrained mirror pixels).

    Raises
    ------
    RuntimeError
        If MASt3R is not installed.
    """
    if not _MAST3R_AVAILABLE:
        raise RuntimeError(
            "MASt3R is required for RCS mask computation but is not installed. "
            "Run: pip install -e third_party/MASt3R"
        )

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gt_arr = _pil_to_uint8_rgb(gt_image)
    mask_arr = _pil_to_binary(mirror_mask)
    H, W = gt_arr.shape[:2]

    scene_arr, mirror_arr = _build_views(gt_arr, mask_arr)

    pts_scene, pts_mirror = _run_mast3r_correspondences(
        scene_arr, mirror_arr, mast3r_model_name, device
    )

    correspondence_mask = np.zeros((H, W), dtype=bool)
    if pts_mirror.shape[0] > 0:
        # Scale coords from MASt3R inference resolution back to original
        scale_x = W / _MAST3R_IMG_SIZE
        scale_y = H / _MAST3R_IMG_SIZE
        xs = np.clip(np.round(pts_mirror[:, 0] * scale_x).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts_mirror[:, 1] * scale_y).astype(int), 0, H - 1)
        # Undo horizontal flip: the mirror view was flipped before inference
        xs = W - 1 - xs
        correspondence_mask[ys, xs] = True
    else:
        logger.warning(
            "compute_rcs_mask: no correspondences found for sample '%s'. "
            "Returning an all-zero mask.",
            mask_stem,
        )

    dilated = _dilate_and_intersect(correspondence_mask, mask_arr, dilation_radius)

    save_path = Path("rcs_masks") / dataset_type / f"{mask_stem}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray((dilated.astype(np.uint8) * 255), mode="L")
    mask_img.save(save_path)
    return save_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pil_to_uint8_rgb(pil: Image.Image) -> np.ndarray:
    """Convert any PIL image to uint8 RGB numpy array (H, W, 3)."""
    if pil.mode not in ("RGB",):
        pil = pil.convert("RGB")
    return np.asarray(pil, dtype=np.uint8)


def _pil_to_binary(pil: Image.Image) -> np.ndarray:
    """Convert a PIL mask image to a boolean numpy array (True = mirror region)."""
    return np.asarray(pil.convert("L"), dtype=np.uint8) > 127


def _build_views(
    image_arr: np.ndarray, mirror_mask_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scene_arr, mirror_arr).

    scene_arr : mirror region zeroed out (Yscene).
    mirror_arr : only the mirror region, horizontally flipped (Ymirror).
    """
    scene = image_arr.copy()
    scene[mirror_mask_arr] = 0

    mirror = image_arr.copy()
    mirror[~mirror_mask_arr] = 0
    mirror = np.fliplr(mirror)

    return scene, mirror


def _load_model(mast3r_model_name: str, device: str) -> "AsymmetricMASt3R":
    """Load (or retrieve cached) MASt3R model."""
    key = (mast3r_model_name, device)
    if key not in _MODEL_CACHE:
        logger.info("Loading MASt3R model '%s' on %s ...", mast3r_model_name, device)
        _MODEL_CACHE[key] = AsymmetricMASt3R.from_pretrained(mast3r_model_name).to(device)
    return _MODEL_CACHE[key]


def _run_mast3r_correspondences(
    scene_arr: np.ndarray,
    mirror_arr: np.ndarray,
    mast3r_model_name: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run MASt3R on the two views and return matched pixel coordinates.

    Returns
    -------
    pts_scene : np.ndarray, shape (N, 2)
        Matched pixel (x, y) coords in the scene view at MASt3R resolution.
    pts_mirror : np.ndarray, shape (N, 2)
        Matched pixel (x, y) coords in the mirror view at MASt3R resolution.
        These coordinates correspond to the *flipped* mirror view; the caller
        is responsible for un-flipping the x axis.
    """
    tmp_dir = tempfile.mkdtemp(prefix="rcs_mast3r_")
    try:
        scene_path = str(Path(tmp_dir) / "scene.png")
        mirror_path = str(Path(tmp_dir) / "mirror.png")
        Image.fromarray(scene_arr).save(scene_path)
        Image.fromarray(mirror_arr).save(mirror_path)

        images = load_images([scene_path, mirror_path], size=_MAST3R_IMG_SIZE, square_ok=True, verbose=False)
        model = _load_model(mast3r_model_name, device)

        output = inference(
            [tuple(images)], model, device, batch_size=1, verbose=False
        )

        # Extract per-pixel descriptors from both views
        desc1 = output["pred1"]["desc"].squeeze(0).detach()  # (H', W', D)
        desc2 = output["pred2"]["desc"].squeeze(0).detach()  # (H', W', D)

        pts_scene, pts_mirror = fast_reciprocal_NNs(
            desc1, desc2,
            subsample_or_initxy1=8,
            device=device,
            dist="dot",
            block_size=2**13,
        )
        # pts_* are (N, 2) arrays of (x, y) at MASt3R inference resolution
        return pts_scene, pts_mirror

    except Exception as exc:
        logger.error("MASt3R inference failed: %s", exc)
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _make_disk_structure(radius: int) -> np.ndarray:
    """Create a 2-D boolean disk structuring element of the given radius."""
    diameter = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x ** 2 + y ** 2 <= radius ** 2


def _dilate_and_intersect(
    correspondence_mask: np.ndarray,
    mirror_mask_arr: np.ndarray,
    dilation_radius: int,
) -> np.ndarray:
    """
    Dilate the correspondence mask and intersect with the mirror mask.

    Parameters
    ----------
    correspondence_mask : np.ndarray, bool (H, W)
        Pixels with valid MASt3R correspondences.
    mirror_mask_arr : np.ndarray, bool (H, W)
        Mirror region mask.
    dilation_radius : int
        Disk radius for dilation. 0 means no dilation.

    Returns
    -------
    np.ndarray, bool (H, W)
    """
    if dilation_radius > 0:
        struct = _make_disk_structure(dilation_radius)
        dilated = binary_dilation(correspondence_mask, structure=struct)
    else:
        dilated = correspondence_mask.copy()
    return dilated & mirror_mask_arr
