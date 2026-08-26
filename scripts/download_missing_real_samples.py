"""
Download the real-dataset samples that are missing from the local cache
under outputs/shift_tolerant_psnr/shift_tolerant_psnr_v1/real_start*/ (only
20 of the 50 available samples were ever fetched there), into a new
real_missing/ directory with the same layout, so existing scripts
(run_jitter_experiment.py, rerun_shift_search_only.py) can process it
unmodified.

CPU/network only — downloads cached PNGs from R2, does not run inference,
LPIPS, or anything GPU-bound. FLUX.2 Klein is skipped (excluded from the
jitter analysis by request).

Usage
-----
    python scripts/download_missing_real_samples.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real_evaluation as _real_eval  # noqa: E402

from fill_my_mirror.storage import R2Client  # noqa: E402

OUTPUT_ROOT = Path("outputs/shift_tolerant_psnr/shift_tolerant_psnr_v1/real_missing")
ALREADY_CACHED_INDICES = {0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 25, 26, 30, 31, 35, 36, 40, 41, 45, 46}
ALL_INDICES = set(range(50))
MISSING_INDICES = sorted(ALL_INDICES - ALREADY_CACHED_INDICES)

MODELS = [m for m in _real_eval.MODELS if m["slug"] != "flux2_klein_vanilla"]
SEEDS = _real_eval.SEEDS
BASE = _real_eval.R2_BASE_PREFIX


def main() -> None:
    r2 = R2Client()
    print(f"Missing indices ({len(MISSING_INDICES)}): {MISSING_INDICES}")
    print(f"Models: {[m['slug'] for m in MODELS]}  Seeds: {SEEDS}")

    n_written = 0
    with tempfile.TemporaryDirectory(prefix="dl_missing_real_") as tmp_str:
        tmp_root = Path(tmp_str)
        for index in MISSING_INDICES:
            gm_dir = tmp_root / str(index)
            gm_dir.mkdir(parents=True, exist_ok=True)

            keys = {
                "gt_image": f"{BASE}/{index}/gt_image.png",
                "full_mirror_mask": f"{BASE}/{index}/generative_refinement_mask.png",
                "constrained_mask": f"{BASE}/{index}/rcs_mask.png",
            }
            if not all(r2.key_exists(k) for k in keys.values()):
                print(f"  index={index}: missing GT/masks in R2, skipping")
                continue

            local_gt = {}
            for field_name, r2_key in keys.items():
                local_path = gm_dir / f"{field_name}.png"
                r2.download_file(r2_key, local_path)
                mode = "RGB" if field_name == "gt_image" else "L"
                local_gt[field_name] = Image.open(local_path).convert(mode)

            for model in MODELS:
                for seed in SEEDS:
                    img_key = model["key_template"].format(index=index, seed=seed)
                    if not r2.key_exists(img_key):
                        continue
                    local_img = gm_dir / f"{model['slug']}_seed_{seed}.png"
                    r2.download_file(img_key, local_img)
                    gen_pil = Image.open(local_img).convert("RGB")

                    seed_dir = OUTPUT_ROOT / model["slug"] / str(index) / f"seed_{seed}"
                    seed_dir.mkdir(parents=True, exist_ok=True)
                    gen_pil.save(seed_dir / "generated.png")
                    local_gt["gt_image"].save(seed_dir / "ground_truth.png")
                    local_gt["full_mirror_mask"].save(seed_dir / "mirror_mask.png")
                    local_gt["constrained_mask"].save(seed_dir / "constrained_mask.png")
                    n_written += 1

            print(f"  index={index}: done")

    print(f"\nWrote {n_written} seed dirs -> {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
