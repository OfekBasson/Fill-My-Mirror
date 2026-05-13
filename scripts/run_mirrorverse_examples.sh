#!/usr/bin/env bash
# Run Fill My Mirror on all MSD-MirrorVerse examples.
# Blacks out the mirror region before passing to the pipeline.
# Results saved to outputs/mirrorverse_examples/<idx>_result.png

set -euo pipefail

cd /home/ofek_basson/FMM-parent_dir/Fill-My-Mirror

source /home/ofek_basson/miniconda3/etc/profile.d/conda.sh

set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=0

SEED=107

DATA_DIR="data/MSD-MirrorVerse examples"
OUTPUT_DIR="outputs/mirrorverse_examples_seed_${SEED}"
MASKED_TMP=$(mktemp -d)

mkdir -p "$OUTPUT_DIR"

cleanup() { rm -rf "$MASKED_TMP"; }
trap cleanup EXIT

for txt_file in "$DATA_DIR"/*.txt; do
    idx=$(basename "$txt_file" .txt)
    prompt=$(cat "$txt_file")

    # Find the jpg and png for this index
    jpg=$(echo "$DATA_DIR/${idx}_"*.jpg)
    mask=$(echo "$DATA_DIR/${idx}_"*.png)

    if [[ ! -f "$jpg" || ! -f "$mask" ]]; then
        echo "Missing files for index $idx, skipping."
        continue
    fi

    # Black out the mirror region: set masked pixels to black
    masked_image="$MASKED_TMP/${idx}_masked.png"
    python - <<PYEOF
from PIL import Image
import numpy as np

img = Image.open("$jpg").convert("RGB")
mask = Image.open("$mask").convert("L")
arr = np.array(img)
m = np.array(mask) > 127
arr[m] = 0
Image.fromarray(arr).save("$masked_image")
PYEOF

    output_path="$OUTPUT_DIR/${idx}_result.png"

    echo "=== Processing index $idx ==="
    echo "Prompt: $prompt"

    python -m fill_my_mirror \
        --image "$masked_image" \
        --mask "$mask" \
        --prompt "$prompt" \
        --output_path "$output_path" \
        --n 13 \
        --t_prime 625 \
        --seed "$SEED"

    echo "Saved: $output_path"
    echo
done

echo "All done. Results in $OUTPUT_DIR"
