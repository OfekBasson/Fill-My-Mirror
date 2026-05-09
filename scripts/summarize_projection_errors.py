"""
Scan every sample dir under a dataset/geom_subdir prefix in R2, check for error.txt,
and write a summary errors.txt at {dataset}/{geom_subdir}/errors.txt.

Summary format (one line per failed index):
    {index}: {ExceptionType}: {message}

Prints a final count: N errors out of M total indices.

Usage:
    python scripts/summarize_projection_errors.py --dataset mirrorbench_v2 --geom-subdir estimated_geometry
    python scripts/summarize_projection_errors.py --dataset blender --geom-subdir gt_geometry
"""

from __future__ import annotations

import argparse
import tempfile
import traceback
from pathlib import Path

from fill_my_mirror.storage import R2Client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["blender", "real", "mirrorbench_v2"])
    parser.add_argument("--geom-subdir", required=True, choices=["gt_geometry", "estimated_geometry"])
    return parser.parse_args()


def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


def main() -> None:
    args = parse_args()
    r2 = R2Client()

    prefix = f"{args.dataset}/{args.geom_subdir}/"
    print(f"Listing keys under {prefix} ...")
    all_keys = r2.list_keys(prefix)

    error_keys = [k for k in all_keys if k.endswith("/error.txt")]
    index_keys = {k.split("/")[2] for k in all_keys if k.split("/")[2].isdigit()}
    total = len(index_keys)

    print(f"Found {total} sample dirs, {len(error_keys)} with error.txt\n")

    errors: dict[int, str] = {}

    for i, key in enumerate(sorted(error_keys, key=lambda k: int(k.split("/")[2]))):
        index = int(key.split("/")[2])
        try:
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / "error.txt"
                r2.download_file(key, local)
                content = local.read_text(errors="replace")

            # Extract just the exception line (last non-empty line of the traceback)
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            error_line = lines[-1] if lines else "unknown error"
            errors[index] = error_line
            print(f"  [{i + 1}/{len(error_keys)}] {index}: {error_line}")
        except Exception:
            print(f"  [{i + 1}/{len(error_keys)}] {index}: could not download error.txt")
            traceback.print_exc()

    # Write summary
    summary_lines = [f"{idx}: {desc}" for idx, desc in sorted(errors.items())]
    summary_text = "\n".join(summary_lines) + "\n" if summary_lines else ""

    summary_key = f"{args.dataset}/{args.geom_subdir}/errors.txt"
    with tempfile.TemporaryDirectory() as tmp:
        summary_path = Path(tmp) / "errors.txt"
        summary_path.write_text(summary_text)
        r2.upload_file(summary_path, summary_key)

    print(f"\nUploaded errors.txt → {summary_key}")
    print(f"\n{len(errors)} errors out of {total} total indices")


if __name__ == "__main__":
    main()
