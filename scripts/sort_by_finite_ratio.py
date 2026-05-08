"""
For each MirrorBench estimated-geometry sample in R2 that has a debug_plane.json,
compute finite_pts / total_pts ratio and sort the projected image into:
    {output_dir}/high_ratio/{index}.png
    {output_dir}/low_ratio/{index}.png

Usage:
    python scripts/sort_by_finite_ratio.py --output-dir /tmp/ratio_sort
    python scripts/sort_by_finite_ratio.py --output-dir /tmp/ratio_sort --threshold 0.01
"""

from __future__ import annotations

import argparse
import json
import tempfile
import traceback
from pathlib import Path

from fill_my_mirror.storage import R2Client

PREFIX = "blender/estimated_geometry"
# PREFIX = "mirrorbench_v2/estimated_geometry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Ratio above which a sample is considered 'high' (default: 0.01 = 1%%)",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    r2 = R2Client()

    high_dir = args.output_dir / "high_ratio"
    low_dir = args.output_dir / "low_ratio"
    high_dir.mkdir(parents=True, exist_ok=True)
    low_dir.mkdir(parents=True, exist_ok=True)

    print("Listing debug_plane.json keys from R2 ...")
    all_keys = r2.list_keys(PREFIX + "/")
    debug_keys = [k for k in all_keys if k.endswith("/debug_plane.json")]
    print(f"Found {len(debug_keys)} samples with debug_plane.json")

    for i, key in enumerate(sorted(debug_keys, key=lambda k: int(k.split("/")[2]))):
        index = key.split("/")[2]
        label = f"[{i + 1}/{len(debug_keys)}] index {index}"

        out_high = high_dir / f"{index}.png"
        out_low = low_dir / f"{index}.png"
        if args.skip_existing and (out_high.exists() or out_low.exists()):
            print(f"{label} — skipping")
            continue

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)

                r2.download_file(key, tmp / "debug_plane.json")
                data = json.loads((tmp / "debug_plane.json").read_text())

                total = data.get("mirror_pts_total", 0)
                finite = data.get("mirror_pts_finite", 0)
                ratio = finite / total if total > 0 else 0.0

                img_key = f"{PREFIX}/{index}/projected_image.png"
                r2.download_file(img_key, tmp / "image.png")

                dest = high_dir if ratio >= args.threshold else low_dir
                import shutil
                shutil.copy(tmp / "image.png", dest / f"{index}.png")

            bucket = "high" if ratio >= args.threshold else "low"
            print(f"{label} — ratio {finite}/{total} = {ratio:.4f} → {bucket}")

        except Exception:
            print(f"{label} — ERROR")
            traceback.print_exc()


if __name__ == "__main__":
    main()
