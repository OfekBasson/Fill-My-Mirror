import tempfile
import warnings
from pathlib import Path
from PIL import Image


def load_hf_sample(repo: str, index: int) -> tuple[str, str, str | None]:
    """Load image and mask from a HF dataset sample.

    Returns:
        (image_path, mask_path, prompt) where paths point to temp PNG files.
    """
    from datasets import load_dataset

    ds = load_dataset(repo, split="test")
    if index < 0 or index >= len(ds):
        raise ValueError(f"--hf-index must be between 0 and {len(ds) - 1}, got {index}")

    sample = ds[index]
    tmp_dir = Path(tempfile.mkdtemp(prefix="fill_my_mirror_hf_"))

    image_path = tmp_dir / "image.png"
    mask_path = tmp_dir / "mask.png"
    sample["image"].save(image_path)
    sample["mask"].save(mask_path)

    prompt = sample.get("caption") or None
    return str(image_path), str(mask_path), prompt


def check_and_fix_aspect_ratio(image_path: str, height: int, width: int) -> int:
    """Check if the requested resolution matches the image aspect ratio.

    If not, keeps height and adjusts width to match, emitting a warning.

    Returns:
        The (possibly corrected) width.
    """
    img = Image.open(image_path)
    img_w, img_h = img.size  # PIL gives (width, height)
    if abs(img_h / img_w - height / width) > 1e-2:
        corrected_width = round(height * img_w / img_h)
        warnings.warn(
            f"Requested resolution {height}x{width} does not match the image aspect ratio "
            f"({img_h}x{img_w}). Adjusting width to {corrected_width}."
        )
        return corrected_width
    return width
