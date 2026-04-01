import warnings
from PIL import Image


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
