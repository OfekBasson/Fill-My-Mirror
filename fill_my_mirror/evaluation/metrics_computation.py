"""
Evaluation metrics computation for mirror inpainting.

Provides ``compute_metrics``, which computes PSNR, SSIM, and LPIPS over two
evaluation regions (full mirror mask and constrained mirror pixels mask) together
with full-image CLIP similarity.

All heavy optional dependencies (scikit-image, lpips, torch, clip) are imported
lazily at module level with graceful fallbacks so that the module can be imported
even in environments where some packages are missing.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from skimage.metrics import structural_similarity as _ski_ssim
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not installed — SSIM will be skipped.")

try:
    import lpips as _lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
    logger.warning("lpips not installed — LPIPS will be skipped.")

try:
    import torch
    import torchvision.transforms as T
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("torch not installed — LPIPS and CLIP will be skipped.")

try:
    import clip as _clip_lib
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False
    logger.warning("openai-clip not installed — CLIP similarity will be skipped.")

# ---------------------------------------------------------------------------
# Module-level lazy singletons (loaded once per process)
# ---------------------------------------------------------------------------

_lpips_net = None
_lpips_device: Optional[str] = None
_clip_model = None
_clip_preprocess = None
_clip_device: Optional[str] = None

# ---------------------------------------------------------------------------
# Input dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GeneratedImage:
    """A single generated (inpainted) image to be evaluated against the GT."""

    name: str
    image: Image.Image


@dataclass
class MetricsInput:
    """
    All inputs needed for a single evaluation run.

    Parameters
    ----------
    gt_image
        The ground-truth reference image.
    generated_images
        One or more generated images to compare against the ground truth.
    full_mirror_mask
        Binary mask covering the entire mirror region (white = mirror).
    constrained_mask
        Binary mask covering only the geometrically-constrained mirror pixels
        (produced by ``compute_rcs_mask`` or ``compute_gt_geometry_constraint_mask``).
    save_path
        Directory path where the output CSV (``metrics.csv``) will be saved.
        The directory is created if it does not exist.
    prompt
        Text prompt used for CLIP image–text similarity. Leave empty to skip CLIP.
    """

    gt_image: Image.Image
    generated_images: list[GeneratedImage]
    full_mirror_mask: Image.Image
    constrained_mask: Image.Image
    save_path: str | Path
    prompt: str = ""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pil_to_rgb01(pil: Image.Image) -> np.ndarray:
    """Convert any PIL image to a float32 numpy array of shape (H, W, 3) in [0, 1]."""
    if pil.mode not in ("RGB",):
        pil = pil.convert("RGB")
    return np.asarray(pil, dtype=np.float32) / 255.0


def _resize_rgb01(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a float32 (H, W, 3) array to ``shape`` = (H_new, W_new)."""
    h, w = shape
    pil = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
    pil = pil.resize((w, h), Image.LANCZOS)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _get_lpips_net_device() -> tuple:
    """
    Lazily initialise the LPIPS AlexNet network.

    Returns
    -------
    (net, device) or (None, None) if lpips / torch are not available.
    """
    global _lpips_net, _lpips_device
    if _lpips_net is not None:
        return _lpips_net, _lpips_device
    if not (_LPIPS_AVAILABLE and _TORCH_AVAILABLE):
        return None, None
    try:
        _lpips_device = "cuda" if torch.cuda.is_available() else "cpu"
        _lpips_net = _lpips_lib.LPIPS(net="alex").to(_lpips_device)
        _lpips_net.eval()
        logger.info("LPIPS AlexNet loaded on %s.", _lpips_device)
    except Exception as exc:
        logger.warning("LPIPS initialisation failed: %s", exc)
        _lpips_net = None
        _lpips_device = None
    return _lpips_net, _lpips_device


def _lpips_distance(
    net,
    device: str,
    arr1: np.ndarray,
    arr2: np.ndarray,
) -> Optional[float]:
    """
    Compute LPIPS between two float32 (H, W, 3) arrays in [0, 1].

    The arrays may have non-mask pixels zeroed by the caller; LPIPS is computed
    over the entire spatial extent (this is an approximation — see docstring of
    ``compute_metrics`` for details).

    Returns ``None`` if computation fails.
    """
    if net is None:
        return None
    try:
        # LPIPS expects tensors in [-1, 1] with shape (1, C, H, W)
        def _to_tensor(arr: np.ndarray) -> "torch.Tensor":
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
            return t * 2.0 - 1.0  # [0,1] → [-1,1]

        t1 = _to_tensor(arr1).to(device)
        t2 = _to_tensor(arr2).to(device)
        with torch.no_grad():
            dist = net(t1, t2)
        return float(dist.item())
    except Exception as exc:
        logger.warning("LPIPS computation failed: %s", exc)
        return None


def _compute_psnr(ref: np.ndarray, arr: np.ndarray, mask: np.ndarray) -> float:
    """PSNR (dB) over the masked region. Returns inf if MSE == 0."""
    diff2 = (arr - ref) ** 2
    mse = float(diff2[mask].mean())
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _prepare_mask(mask_pil: Image.Image, shape: tuple[int, int]) -> np.ndarray:
    """Resize a mask image (nearest-neighbour) to ``shape`` = (H, W) and threshold to bool."""
    h, w = shape
    arr = np.asarray(mask_pil.convert("L").resize((w, h), Image.NEAREST), dtype=np.uint8)
    return arr > 127


def _shift_full(arr: np.ndarray, dy: int, dx: int, pad_value: Optional[bool] = None) -> np.ndarray:
    """
    Translate a full (H, W[, C]) array by (dy, dx) pixels (positive dy = down,
    positive dx = right). The border the shift exposes is filled by
    edge-replication (``pad_value=None`` — appropriate for image content) or a
    constant (``pad_value=False`` — appropriate for boolean masks, so a
    shifted-in border can never read as "inside the mask").
    """
    if dy == 0 and dx == 0:
        return arr
    h, w = arr.shape[:2]
    pad = max(abs(dy), abs(dx))
    pad_width = ((pad, pad), (pad, pad)) + ((0, 0),) * (arr.ndim - 2)
    if pad_value is None:
        padded = np.pad(arr, pad_width, mode="edge")
    else:
        padded = np.pad(arr, pad_width, mode="constant", constant_values=pad_value)
    return padded[pad - dy:pad - dy + h, pad - dx:pad - dx + w]


def compute_jitter_psnr(
    ref: np.ndarray,
    arr: np.ndarray,
    mask: np.ndarray,
    jitter_radius: int = 3,
) -> dict:
    """
    Per-pixel local-jitter-tolerant PSNR/MSE:

        MSE_jitter = mean_{(x,y) in mask} min_{(a,b) in [-J,J]^2} ||ref(x,y) - arr(x+a,y+b)||^2

    Each pixel picks its own best local offset independently, within a
    small ``jitter_radius`` = J — there is no shared, whole-region offset.

    CAVEAT (this is why it's implemented for comparison, not as a
    recommended metric): with no requirement that neighbouring pixels agree
    on a similar offset, this has no coherence constraint at all — a real
    misregistration moves the whole reflection together, but this metric
    doesn't check for that. In any locally-smooth or locally-textured
    region, nearby pixels already resemble each other, so taking a min over
    a (2J+1)x(2J+1) neighbourhood *for every pixel independently* will show
    a spurious improvement over plain MSE even between two images with no
    real relationship — and more so for blurrier/smoother content, since
    blur increases local self-similarity. See the decoy test in this
    module's test suite for a direct demonstration.

    Same "mask it" discipline as the rest of this module: a candidate
    source pixel (x+a, y+b) only counts if it's also inside ``mask``
    (prevents background/frame content from leaking in) — though because
    the search radius is tiny, this matters far less here than the
    coherence problem above. Zero offset (a=b=0) is always a valid
    candidate for every masked pixel (source == destination mask there), so
    no pixel is ever fully excluded.

    Returns a dict with ``mse_jitter`` and ``psnr_jitter`` (inf if MSE is 0),
    plus ``mse_plain``/``psnr_plain`` (jitter_radius=0, i.e. plain masked
    MSE/PSNR over the same pixels) for direct side-by-side comparison.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise ValueError("mask is empty — no mirror pixels to evaluate.")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    h, w = mask.shape
    J = jitter_radius
    cy0, cy1 = max(0, y0 - J), min(h, y1 + J)
    cx0, cx1 = max(0, x0 - J), min(w, x1 + J)
    ref_c = ref[cy0:cy1, cx0:cx1]
    arr_c = arr[cy0:cy1, cx0:cx1]
    mask_c = mask[cy0:cy1, cx0:cx1]

    dest_mask = mask_c
    min_diff2 = None
    plain_diff2 = ((ref_c - arr_c) ** 2).sum(axis=-1)

    for a in range(-J, J + 1):
        for b in range(-J, J + 1):
            arr_shifted = _shift_full(arr_c, a, b, pad_value=None)
            source_mask = _shift_full(mask_c, a, b, pad_value=False)
            valid = dest_mask & source_mask
            diff2 = ((ref_c - arr_shifted) ** 2).sum(axis=-1)
            # Invalid candidates must never win the per-pixel min — push them to +inf.
            diff2 = np.where(valid, diff2, np.inf)
            min_diff2 = diff2 if min_diff2 is None else np.minimum(min_diff2, diff2)

    mse_jitter = float(min_diff2[dest_mask].mean() / ref_c.shape[-1])  # per-channel mean, matching _compute_psnr
    mse_plain = float(plain_diff2[dest_mask].mean() / ref_c.shape[-1])

    def _psnr(mse: float) -> float:
        return float("inf") if mse <= 0.0 else 10.0 * math.log10(1.0 / mse)

    return {
        "mse_jitter": mse_jitter, "psnr_jitter": _psnr(mse_jitter),
        "mse_plain": mse_plain, "psnr_plain": _psnr(mse_plain),
    }


def _compute_ssim(ref: np.ndarray, arr: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """SSIM averaged over masked pixels. Returns None if skimage is unavailable."""
    if not _SKIMAGE_AVAILABLE:
        return None
    try:
        _, ssim_map = _ski_ssim(
            ref, arr, data_range=1.0, channel_axis=2, full=True
        )
    except TypeError:
        # Older skimage versions use multichannel instead of channel_axis
        _, ssim_map = _ski_ssim(
            ref, arr, data_range=1.0, multichannel=True, full=True
        )
    return float(ssim_map[mask].mean())


def _compute_clip_similarity(
    img_pil: Image.Image,
    prompt: str,
    device: Optional[str] = None,
) -> Optional[float]:
    """
    Compute CLIP ViT-B/32 cosine similarity between an image and a text prompt.

    Parameters
    ----------
    img_pil : PIL.Image
        Full image (not masked) to encode.
    prompt : str
        Text description to compare against.
    device : str, optional
        Torch device. Defaults to CUDA if available.

    Returns
    -------
    float or None
        Cosine similarity in [-1, 1], or ``None`` if CLIP is unavailable or
        an error occurs.
    """
    global _clip_model, _clip_preprocess, _clip_device
    if not (_CLIP_AVAILABLE and _TORCH_AVAILABLE):
        return None
    try:
        if _clip_model is None:
            _clip_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            _clip_model, _clip_preprocess = _clip_lib.load(
                "ViT-B/32", device=_clip_device, jit=False
            )
            _clip_model.eval()

        dev = _clip_device
        image_input = _clip_preprocess(img_pil).unsqueeze(0).to(dev)
        with torch.no_grad():
            text_tokens = _clip_lib.tokenize([prompt]).to(dev)
            img_feat = _clip_model.encode_image(image_input)
            txt_feat = _clip_model.encode_text(text_tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ txt_feat.T).squeeze().item()
        return float(sim)
    except Exception as exc:
        logger.warning("CLIP similarity computation failed: %s", exc)
        return None


def compute_metrics(metrics_input: MetricsInput) -> pd.DataFrame:
    """
    Compute PSNR, SSIM, LPIPS, and CLIP metrics for a list of generated images.

    Metrics are computed for two regions:

    - **Full mirror mask** – the entire mirror region.
    - **Constrained mask** – only the geometrically-constrained mirror pixels
      (produced by RCS or GT geometry projection).

    Additionally, CLIP image–text cosine similarity is computed over the full
    image when ``metrics_input.prompt`` is non-empty.

    .. note::
        LPIPS is computed on the full spatial extent of the image after zeroing
        non-mask pixels.  This is an approximation: LPIPS uses a deep network
        that incorporates context beyond the masked region.

    Parameters
    ----------
    metrics_input : MetricsInput
        Evaluation inputs.

    Returns
    -------
    pd.DataFrame
        One row per generated image.  Columns:

        ``name``,
        ``psnr_full_mirror``, ``ssim_full_mirror``, ``lpips_full_mirror``,
        ``psnr_constrained``, ``ssim_constrained``, ``lpips_constrained``,
        ``clip_similarity``
    """
    ref = _pil_to_rgb01(metrics_input.gt_image)

    full_mask = _prepare_mask(metrics_input.full_mirror_mask, ref.shape[:2])
    constrained_mask = _prepare_mask(metrics_input.constrained_mask, ref.shape[:2])

    if full_mask.sum() == 0:
        raise ValueError("Full mirror mask is empty (no pixels above threshold).")
    if constrained_mask.sum() == 0:
        logger.warning(
            "Constrained mask is empty — constrained-region metrics will be NaN."
        )

    lpips_net, lpips_device = _get_lpips_net_device()

    rows = []
    for gen in metrics_input.generated_images:
        arr = _pil_to_rgb01(gen.image)
        if arr.shape[:2] != ref.shape[:2]:
            arr = _resize_rgb01(arr, (ref.shape[0], ref.shape[1]))

        # --- Full mirror mask metrics ---
        psnr_full = _compute_psnr(ref, arr, full_mask)
        ssim_full = _compute_ssim(ref, arr, full_mask)
        ref_masked_full = ref * full_mask[..., None]
        arr_masked_full = arr * full_mask[..., None]
        lpips_full = _lpips_distance(lpips_net, lpips_device, ref_masked_full, arr_masked_full)

        # --- Constrained mask metrics ---
        if constrained_mask.sum() > 0:
            psnr_constr = _compute_psnr(ref, arr, constrained_mask)
            ssim_constr = _compute_ssim(ref, arr, constrained_mask)
            ref_masked_constr = ref * constrained_mask[..., None]
            arr_masked_constr = arr * constrained_mask[..., None]
            lpips_constr = _lpips_distance(
                lpips_net, lpips_device, ref_masked_constr, arr_masked_constr
            )
        else:
            psnr_constr = float("nan")
            ssim_constr = None
            lpips_constr = None

        # --- CLIP similarity (full image) ---
        clip_sim: Optional[float] = None
        if metrics_input.prompt:
            clip_sim = _compute_clip_similarity(gen.image, metrics_input.prompt)

        row: dict = {"name": gen.name}
        row["psnr_full_mirror"] = psnr_full
        if ssim_full is not None:
            row["ssim_full_mirror"] = ssim_full
        if lpips_full is not None:
            row["lpips_full_mirror"] = lpips_full
        row["psnr_constrained"] = psnr_constr
        if ssim_constr is not None:
            row["ssim_constrained"] = ssim_constr
        if lpips_constr is not None:
            row["lpips_constrained"] = lpips_constr
        if clip_sim is not None:
            row["clip_similarity"] = clip_sim

        rows.append(row)

    df = pd.DataFrame(rows)

    save_path = Path(metrics_input.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    csv_path = save_path / "metrics.csv"
    df.to_csv(csv_path, index=False)

    return df
