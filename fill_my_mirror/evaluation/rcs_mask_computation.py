"""
Reflection Consistency Score (RCS) mask computation using MASt3R.

Identifies mirror pixels that are geometrically constrained by the visible scene
by finding dense pixel correspondences between the scene view and a horizontally
flipped mirror view (paper Section 4.3).

Requires MASt3R to be set up (see third_party/MASt3R/README.md):
    pip install -r third_party/MASt3R/requirements.txt
    pip install -r third_party/MASt3R/dust3r/requirements.txt
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# MASt3R imports are deferred until first use to avoid polluting sys.path at
# module load time (dust3r adds its croco/ dir to sys.path, which would shadow
# the HuggingFace `datasets` package otherwise).
_MAST3R_AVAILABLE: Optional[bool] = None  # None = not yet attempted


def _ensure_mast3r() -> bool:
    """Lazily inject MASt3R/dust3r into sys.path and attempt to import them."""
    global _MAST3R_AVAILABLE
    if _MAST3R_AVAILABLE is not None:
        return _MAST3R_AVAILABLE

    mast3r_root = Path(__file__).resolve().parents[2] / "third_party" / "MASt3R"
    for p in [str(mast3r_root), str(mast3r_root / "dust3r")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        import mast3r.model  # noqa: F401
        import mast3r.fast_nn  # noqa: F401
        import dust3r.inference  # noqa: F401
        import dust3r.utils.image  # noqa: F401
        _MAST3R_AVAILABLE = True
    except ImportError:
        _MAST3R_AVAILABLE = False
        logger.warning(
            "MASt3R dependencies not found. RCS mask computation will be unavailable. "
            "Install with: pip install -r third_party/MASt3R/requirements.txt "
            "-r third_party/MASt3R/dust3r/requirements.txt"
        )
    return _MAST3R_AVAILABLE

# Module-level model cache: avoids reloading on every call in a batch run.
_MODEL_CACHE: dict[tuple[str, str], object] = {}

_MAST3R_IMG_SIZE = 512


def compute_rcs_mask(
    gt_image: Image.Image,
    mirror_mask: Image.Image,
    dataset_type: str,
    mask_stem: str,
    mast3r_model_name: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    device: Optional[str] = None,
    dilation_radius: int = 5,
    dilation_iterations: int = 1,
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
        Radius of the circular dilation kernel (kernel size = ``(2*r+1) × (2*r+1)``).
        Set to 0 to disable dilation.
    dilation_iterations : int
        Number of dilation iterations. Default is 1.

    Returns
    -------
    Path
        Path to the saved binary mask PNG (white = constrained mirror pixels).

    Raises
    ------
    RuntimeError
        If MASt3R is not installed.
    """
    if not _ensure_mast3r():
        raise RuntimeError(
            "MASt3R is required for RCS mask computation but its dependencies are missing. "
            "Run: pip install -r third_party/MASt3R/requirements.txt "
            "-r third_party/MASt3R/dust3r/requirements.txt"
        )

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gt_arr = _pil_to_uint8_rgb(gt_image)
    mask_arr = _pil_to_binary(mirror_mask)
    H, W = gt_arr.shape[:2]

    scene_arr, _ = _build_views(gt_arr, mask_arr)

    pts_scene, pts_mirror, use_rot180 = _run_mast3r_best_correspondences(
        scene_arr, gt_arr, mask_arr, mast3r_model_name, device
    )

    correspondence_mask = np.zeros((H, W), dtype=bool)
    if pts_mirror.shape[0] > 0:
        # Scale coords from MASt3R inference resolution back to original
        scale_x = W / _MAST3R_IMG_SIZE
        scale_y = H / _MAST3R_IMG_SIZE
        xs = np.clip(np.round(pts_mirror[:, 0] * scale_x).astype(int), 0, W - 1)
        ys = np.clip(np.round(pts_mirror[:, 1] * scale_y).astype(int), 0, H - 1)
        # Undo the transform applied to the mirror view before MASt3R inference
        xs = W - 1 - xs
        if use_rot180:
            ys = H - 1 - ys
        correspondence_mask[ys, xs] = True
    else:
        logger.warning(
            "compute_rcs_mask: no correspondences found for sample '%s'. "
            "Returning an all-zero mask.",
            mask_stem,
        )

    dilated = _dilate_and_intersect(correspondence_mask, mask_arr, dilation_radius, iterations=dilation_iterations)

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


def _build_mirror_rot180(
    image_arr: np.ndarray, mirror_mask_arr: np.ndarray
) -> np.ndarray:
    """Return the mirror view rotated 180° (alternative to horizontal flip)."""
    mirror = image_arr.copy()
    mirror[~mirror_mask_arr] = 0
    return np.rot90(mirror, 2)


def _run_mast3r_best_correspondences(
    scene_arr: np.ndarray,
    image_arr: np.ndarray,
    mirror_mask_arr: np.ndarray,
    mast3r_model_name: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Run MASt3R with both mirror-view variants; return the one with more matches.

    Tries horizontal flip (current default) and 180° rotation. Chooses whichever
    produces more reciprocal correspondences.

    Returns
    -------
    pts_scene : np.ndarray, shape (N, 2)
    pts_mirror : np.ndarray, shape (N, 2)  — coords in the *chosen* mirror view space
    use_rot180 : bool — True if the 180° variant was chosen
    """
    _, mirror_hflip = _build_views(image_arr, mirror_mask_arr)
    mirror_rot180 = _build_mirror_rot180(image_arr, mirror_mask_arr)

    pts_scene_hflip, pts_mirror_hflip = _run_mast3r_correspondences(
        scene_arr, mirror_hflip, mast3r_model_name, device
    )
    pts_scene_rot180, pts_mirror_rot180 = _run_mast3r_correspondences(
        scene_arr, mirror_rot180, mast3r_model_name, device
    )

    n_hflip = pts_mirror_hflip.shape[0]
    n_rot180 = pts_mirror_rot180.shape[0]
    logger.info(
        "MASt3R correspondences — hflip: %d, rot180: %d → choosing %s",
        n_hflip, n_rot180, "rot180" if n_rot180 > n_hflip else "hflip",
    )

    if n_rot180 > n_hflip:
        return pts_scene_rot180, pts_mirror_rot180, True
    return pts_scene_hflip, pts_mirror_hflip, False


def _load_model(mast3r_model_name: str, device: str) -> object:
    """Load (or retrieve cached) MASt3R model."""
    from mast3r.model import AsymmetricMASt3R
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

        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from mast3r.fast_nn import fast_reciprocal_NNs

        images = load_images([scene_path, mirror_path], size=_MAST3R_IMG_SIZE, square_ok=True, verbose=False)
        model = _load_model(mast3r_model_name, device)

        output = inference(
            [tuple(images)], model, device, batch_size=1, verbose=False
        )

        view1, view2 = output["view1"], output["view2"]

        # Extract per-pixel descriptors from both views
        desc1 = output["pred1"]["desc"].squeeze(0).detach()  # (H', W', D)
        desc2 = output["pred2"]["desc"].squeeze(0).detach()  # (H', W', D)

        # subsample_or_initxy1=1 queries every pixel → maximum correspondence density
        pts_scene, pts_mirror = fast_reciprocal_NNs(
            desc1, desc2,
            subsample_or_initxy1=1,
            device=device,
            dist="dot",
            block_size=2**13,
        )

        # Filter out matches on the 3-pixel border (unreliable descriptor region)
        H0, W0 = view1["true_shape"][0]
        H1, W1 = view2["true_shape"][0]
        valid0 = (
            (pts_scene[:, 0] >= 3) & (pts_scene[:, 0] < int(W0) - 3) &
            (pts_scene[:, 1] >= 3) & (pts_scene[:, 1] < int(H0) - 3)
        )
        valid1 = (
            (pts_mirror[:, 0] >= 3) & (pts_mirror[:, 0] < int(W1) - 3) &
            (pts_mirror[:, 1] >= 3) & (pts_mirror[:, 1] < int(H1) - 3)
        )
        valid = valid0 & valid1
        pts_scene, pts_mirror = pts_scene[valid], pts_mirror[valid]

        # pts_* are (N, 2) arrays of (x, y) at MASt3R inference resolution
        return pts_scene, pts_mirror

    except Exception as exc:
        logger.error("MASt3R inference failed: %s", exc)
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _dilate_and_intersect(
    correspondence_mask: np.ndarray,
    mirror_mask_arr: np.ndarray,
    dilation_radius: int,
    iterations: int = 2,
) -> np.ndarray:
    """
    Dilate the correspondence mask and intersect with the mirror mask.

    Uses a circular kernel (``cv2.MORPH_ELLIPSE``).

    Parameters
    ----------
    correspondence_mask : np.ndarray, bool (H, W)
        Pixels with valid MASt3R correspondences.
    mirror_mask_arr : np.ndarray, bool (H, W)
        Mirror region mask.
    dilation_radius : int
        Radius of the circular kernel (kernel size = ``(2*r+1) × (2*r+1)``).
        Set to 0 to disable dilation.
    iterations : int
        Number of dilation iterations. Default is 2.

    Returns
    -------
    np.ndarray, bool (H, W)
    """
    if dilation_radius > 0:
        kernel_size = 2 * dilation_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_uint8 = correspondence_mask.astype(np.uint8) * 255
        dilated_uint8 = cv2.dilate(mask_uint8, kernel, iterations=iterations)
        dilated = dilated_uint8 > 127
    else:
        dilated = correspondence_mask.copy()
    return dilated & mirror_mask_arr
