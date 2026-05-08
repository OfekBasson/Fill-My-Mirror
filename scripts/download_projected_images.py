"""
Download projected_image.png from every sample in R2 into a flat local directory.

Usage
-----
    python scripts/download_projected_images.py \\
        --dataset mirrorbench_v2 \\
        --geometry gt_geometry \\
        --output-dir /tmp/projected_images

Files are saved as {output_dir}/{index}.png.
"""

import argparse
from pathlib import Path

from fill_my_mirror.storage import R2Client


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download projected images from R2 to a local directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", required=True, choices=["real", "blender", "mirrorbench_v2"],
        help="Dataset name (matches the R2 prefix).",
    )
    parser.add_argument(
        "--geometry", required=True, choices=["gt_geometry", "estimated_geometry"],
        help="Geometry subdirectory to download from.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=str,
        help="Local directory to save images into.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    r2 = R2Client()
    prefix = f"{args.dataset}/{args.geometry}/"
    all_keys = r2.list_keys(prefix)
    projected_keys = [k for k in all_keys if k.endswith("/projected_image.png")]

    if not projected_keys:
        print(f"No projected_image.png files found under R2:{prefix}")
        return

    print(f"Found {len(projected_keys)} projected images. Downloading to {output_dir} ...")

    for key in sorted(projected_keys):
        index = key.split("/")[-2]
        local_path = output_dir / f"{index}.png"
        if local_path.exists():
            print(f"  {index}.png — already exists, skipping")
            continue
        r2.download_file(key, local_path)
        print(f"  {index}.png")

    print(f"\nDone. {len(projected_keys)} images in {output_dir}")


if __name__ == "__main__":
    main()
