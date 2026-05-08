"""
Download and extract the MirrorBench V2 (SynMirrorV2) dataset.

Downloads all tar archives from HuggingFace into data/mirrorbench_v2/ and
extracts them in place.

Usage
-----
Download and extract everything (default):

    python scripts/download_mirrorbench_v2.py

Download a single batch archive only:

    python scripts/download_mirrorbench_v2.py --batch 0

Skip extraction (download only):

    python scripts/download_mirrorbench_v2.py --no-extract
"""

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from fill_my_mirror.loaders import MIRRORBENCH_V2_HF_REPO, MIRRORBENCH_V2_DATA_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract the MirrorBench V2 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        metavar="N",
        help="Download only batch_N.tar instead of all archives.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        default=False,
        help="Download archives but do not extract them.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MIRRORBENCH_V2_DATA_ROOT),
        help=f"Directory to download and extract into (default: {MIRRORBENCH_V2_DATA_ROOT}).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch is not None:
        filename = f"batch_{args.batch}.tar"
        print(f"Downloading {filename} ...")
        local_path = hf_hub_download(
            repo_id=MIRRORBENCH_V2_HF_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=str(output_dir),
        )
        tar_files = [Path(local_path)]
    else:
        print("Downloading all tar archives ...")
        snapshot_download(
            repo_id=MIRRORBENCH_V2_HF_REPO,
            repo_type="dataset",
            allow_patterns=["*.tar"],
            local_dir=str(output_dir),
        )
        tar_files = sorted(output_dir.glob("*.tar"))

    if args.no_extract:
        print(f"Skipping extraction. Archives are in {output_dir}/")
        return

    for tar_path in tar_files:
        print(f"Extracting {tar_path.name} ...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(path=output_dir)

    print(f"\nDone. Dataset extracted to {output_dir}/")


if __name__ == "__main__":
    main()
