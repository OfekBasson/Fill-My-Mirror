# RCS Dilation Radius Sweep

Sweeps `dilation_radius` values for the RCS mask and compares each to the
ground-truth geometry constraint mask computed from the Blender dataset.

MASt3R is run **twice per sample** — once with the mirror view horizontally
flipped (`hflip`) and once rotated 180° (`rot180`). The correspondence points
from both runs are unioned into a single mask, which is then used as the basis
for dilation. Dilation is applied separately for each radius, so the expensive
network inference is not repeated per radius.

The best (radius, iterations) pair is selected by Fβ. The default β=0.5 weights
precision 4× over recall; use `--beta 0.25` for 16× (even more precision-focused).
β can be changed after the fact with `--summary-only` without re-running dilation.

## Usage

**Full run** (MASt3R + dilation sweep):

```bash
conda run -n fill-my-mirror python experiments/rcs_dilation_sweep/sweep.py \
    --config configs/config.yaml \
    --indices 0 1 2 3 4 \
    --radii 0 1 2 3 4 5 6 \
    --iterations 1 2 3 \
    --output-dir experiments/rcs_dilation_sweep/results
```

**Re-dilate only** (reuse saved correspondence masks, skip MASt3R):

```bash
# 1. Delete old per-radius outputs and summaries
find experiments/rcs_dilation_sweep/results/sample_* \
    -name "rcs_r*.png" -o -name "overlay_r*.png" \
    -o -name "rcs_r*_i*.png" -o -name "overlay_r*_i*.png" | xargs rm -f
rm -f experiments/rcs_dilation_sweep/results/metrics.csv \
      experiments/rcs_dilation_sweep/results/summary.csv \
      experiments/rcs_dilation_sweep/results/*.png

# 2. Re-run dilation with new radii and iteration counts
conda run -n fill-my-mirror python experiments/rcs_dilation_sweep/sweep.py \
    --config configs/config.yaml \
    --radii $(seq 0 20) \
    --iterations 1 2 3 \
    --output-dir experiments/rcs_dilation_sweep/results \
    --redilate
```

**Summary-only** (recompute Fβ and plots from existing `metrics.csv`, no dilation):

```bash
conda run -n fill-my-mirror python experiments/rcs_dilation_sweep/sweep.py \
    --output-dir experiments/rcs_dilation_sweep/results \
    --beta 0.25 \
    --summary-only
```

`--summary-only` loads the existing `metrics.csv`, recomputes `fbeta` with the
given `--beta`, and overwrites `summary.csv` and all plots. No dilation,
no MASt3R, no per-sample image loading. Use this to experiment with different β
values instantly.

---

`--redilate` loads `sample_{i}/combined_correspondence_mask.png` and
`sample_{i}/gt_mask.png` saved by a previous full run, then re-runs only the
dilation step for every combination of `--radii` and `--iterations`. All
per-radius masks, overlays, and summary outputs are overwritten. MASt3R is
never loaded. Requires that the full run has been completed first.

`--iterations` sets how many times the dilation kernel is applied per radius.
Multiple values can be provided to sweep over them (e.g. `--iterations 1 2 3`).
Per-(radius, iterations) outputs are saved as `rcs_r{r}_i{iters}.png` and
`overlay_r{r}_i{iters}.png`. Summary plots are generated separately for each
iterations value.

`--beta` controls the Fβ score used to select the best (radius, iterations):
- `0.5` (default): precision weighted 4× over recall
- `0.25`: precision weighted 16× over recall

Omit `--indices` to run on the full Blender dataset.

## Outputs (`results/`)

| Path | Description |
|------|-------------|
| `sample_{i}/gt_mask.png` | GT geometry constraint mask (binary) |
| `sample_{i}/overlay_gt.png` | GT mask overlaid on original image |
| `sample_{i}/hflip/view_scene.png` | Scene view fed to MASt3R (hflip run) |
| `sample_{i}/hflip/view_mirror.png` | Mirror view fed to MASt3R (hflip run) |
| `sample_{i}/hflip/correspondence_mask.png` | Raw correspondence mask for hflip |
| `sample_{i}/hflip/overlay_correspondence.png` | hflip correspondence mask overlaid on image |
| `sample_{i}/rot180/...` | Same outputs for the rot180 run |
| `sample_{i}/combined_correspondence_mask.png` | Union of hflip and rot180 correspondence masks |
| `sample_{i}/overlay_combined_correspondence.png` | Combined mask overlaid on image |
| `sample_{i}/rcs_r{r}_i{iters}.png` | RCS mask at radius `r`, `iters` iterations (binary) |
| `sample_{i}/overlay_r{r}_i{iters}.png` | RCS mask overlaid on image (P/R/Fβ in title) |
| `metrics.csv` | Per-sample per-(radius, iterations) precision, recall, F1, Fβ |
| `summary.csv` | Mean precision, recall, F1, Fβ per (radius, iterations) across all samples |
| `precision_recall_i{iters}.png` | PR curve for each iterations value |
| `precision_vs_radius_i{iters}.png` | Precision vs radius for each iterations value |
| `recall_vs_radius_i{iters}.png` | Recall vs radius for each iterations value |
| `f1_vs_radius_i{iters}.png` | F1 vs radius for each iterations value |
| `fbeta_vs_radius_i{iters}.png` | Fβ vs radius for each iterations value (best highlighted) |

The script prints the suggested `(dilation_radius, iterations)` pair with the highest mean Fβ.

## Requirements

- Blender installed at the path specified in `configs/config.yaml`
- MASt3R set up: `pip install -r third_party/MASt3R/requirements.txt -r third_party/MASt3R/dust3r/requirements.txt`
