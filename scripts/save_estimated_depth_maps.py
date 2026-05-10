"""
Save estimated depth maps for mirrorbench_v2/estimated_geometry samples in R2.

For each index, downloads original_image.png from R2, runs MoGe inference,
saves the depth as a 16-bit PNG, and uploads it back to R2.

Example
-------
    python scripts/save_estimated_depth_maps.py --start-index 0 --end-index 100 --skip-existing
"""

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from fill_my_mirror.storage import R2Client

R2_DATASET_PREFIX = "mirrorbench_v2/estimated_geometry"
DEFAULT_CONFIG_PATH = Path("configs/config.yaml")

logger = logging.getLogger(__name__)


def _save_depth_uint16(depth: np.ndarray, path: Path) -> None:
    d_min, d_max = float(depth.min()), float(depth.max())
    scale = d_max - d_min
    if scale < 1e-8:
        depth_uint16 = np.zeros_like(depth, dtype=np.uint16)
    else:
        depth_uint16 = ((depth - d_min) / scale * 65535).astype(np.uint16)
    Image.fromarray(depth_uint16).save(path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run MoGe on R2-stored original images and upload depth maps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, required=True,
                        help="Last index to process (exclusive).")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="Skip indices whose depth_map.png already exists in R2.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    geometry_model_name = config.get("geometry_model_name", "Ruicheng/moge-2-vitl-normal")
    if not geometry_model_name.startswith("Ruicheng/moge"):
        raise ValueError(
            f"This script requires a MoGe geometry model, got: {geometry_model_name}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from moge.model.v2 import MoGeModel
    logger.info("Loading MoGe model: %s", geometry_model_name)
    model = MoGeModel.from_pretrained(geometry_model_name).to(device)
    model.eval()

    r2 = R2Client()
    indices = range(args.start_index, args.end_index)
    total = len(indices)

    for i, index in enumerate(indices):
        label = f"[{i + 1}/{total}] index {index}"
        r2_depth_key = f"{R2_DATASET_PREFIX}/{index}/depth_map.png"
        r2_image_key = f"{R2_DATASET_PREFIX}/{index}/original_image.png"

        if args.skip_existing and r2.key_exists(r2_depth_key):
            logger.info("%s — skipping (already in R2)", label)
            continue

        if not r2.key_exists(r2_image_key):
            logger.warning("%s — original_image.png not found in R2, skipping", label)
            continue

        with tempfile.TemporaryDirectory(prefix="save_depth_est_") as tmp_str:
            tmp_dir = Path(tmp_str)
            image_path = tmp_dir / "original_image.png"
            depth_path = tmp_dir / "depth_map.png"

            logger.info("%s — downloading original_image.png", label)
            r2.download_file(r2_image_key, image_path)

            import cv2
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = (
                torch.tensor(image / 255.0, dtype=torch.float32)
                .permute(2, 0, 1)
                .to(device)
            )

            logger.info("%s — running MoGe inference", label)
            with torch.inference_mode():
                output = model.infer(image_tensor, resolution_level=9, apply_mask=True)

            depth = output["depth"].cpu().numpy().astype(np.float32)
            _save_depth_uint16(depth, depth_path)

            logger.info("%s — uploading depth_map.png", label)
            r2.upload_file(depth_path, r2_depth_key)

        logger.info("%s — done", label)


if __name__ == "__main__":
    main()
