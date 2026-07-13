"""
Vanilla FLUX.1 Fill Timing Benchmark
=====================================
Runs plain FLUX.1-Fill-dev inpainting (no geometry estimation, no Blender
projection, no dual-mask/projection components) on the real images, timing
the inpainting step and recording peak GPU memory usage.

Input assets are downloaded from R2 under real/estimated_geometry/<idx>/:
  original_image.png
  generative_refinement_mask.png
  prompt.json

Outputs per sample:
  <output_dir>/sample_<idx>/result.png
  <output_dir>/sample_<idx>/timing.json

Aggregated outputs:
  <output_dir>/timing_summary.csv
  <output_dir>/timing_summary.json

Usage
-----
    # Quick test on 2 samples:
    conda run -n fill-my-mirror python scripts/benchmark_pipeline_timing_vanilla_flux_fill.py --indices 0 1 --output-dir outputs/timing_benchmark_vanilla_test

    # Full run (all 50 real images):
    conda run -n fill-my-mirror python scripts/benchmark_pipeline_timing_vanilla_flux_fill.py \\
        --output-dir outputs/timing_benchmark_vanilla --skip-existing
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

from fill_my_mirror.loaders import EstimatedGeometrySample
from fill_my_mirror.storage import R2Client
from fill_my_mirror.utils import check_and_fix_aspect_ratio

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
R2_PROJ_PREFIX = "real/estimated_geometry"
N_REAL_IMAGES = 50
INPAINTING_MODEL_NAME = "black-forest-labs/FLUX.1-Fill-dev"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _gpu_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def load_vanilla_pipeline(gpu_id: int = 0) -> FluxFillPipeline:
    pipe = FluxFillPipeline.from_pretrained(
        INPAINTING_MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    return pipe


def benchmark_sample(
    idx: int,
    r2: R2Client,
    pipe: FluxFillPipeline,
    config: dict,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    sample_dir = output_dir / f"sample_{idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix=f"fmm_bench_vanilla_{idx}_"))
    t0 = time.perf_counter()
    try:
        # Download inputs from R2
        r2.download_file(f"{R2_PROJ_PREFIX}/{idx}/original_image.png", tmp / "image.png")
        r2.download_file(f"{R2_PROJ_PREFIX}/{idx}/generative_refinement_mask.png", tmp / "mask.png")
        try:
            r2.download_file(f"{R2_PROJ_PREFIX}/{idx}/prompt.json", tmp / "prompt.json")
            prompt = json.loads((tmp / "prompt.json").read_text()).get("prompt", "").strip()
        except Exception:
            prompt = ""
        if not prompt:
            prompt = config["prompt"]

        sample = EstimatedGeometrySample(
            image_path=str(tmp / "image.png"),
            mask_path=str(tmp / "mask.png"),
            prompt=prompt,
        )
        width = check_and_fix_aspect_ratio(sample.image_path, int(args.height), int(args.width))

        image = load_image(sample.image_path).convert("RGB")
        mask = load_image(sample.mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size)

        torch.cuda.reset_peak_memory_stats()

        generator = torch.Generator("cuda").manual_seed(args.seed)

        # Vanilla FLUX.1 Fill inpainting (no geometry, no projection, no dual mask)
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            height=int(args.height),
            width=width,
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            max_sequence_length=args.max_sequence_length,
        )
        t1 = time.perf_counter()
        inpainting_time_s = t1 - t0
        inpainting_peak_gpu_gb = _gpu_mb()

        final_image = result.images[0]
        final_image.save(sample_dir / "result.png")

        result = {
            "index": idx,
            "inpainting_time_s": round(inpainting_time_s, 2),
            "total_time_s": round(t1 - t0, 2),
            "inpainting_peak_gpu_gb": round(inpainting_peak_gpu_gb, 2),
            "overall_peak_gpu_gb": round(inpainting_peak_gpu_gb, 2),
            "error": False,
        }
        logger.info(
            "[%d] inpaint=%.1fs  total=%.1fs  peak_gpu=%.1fGB",
            idx,
            inpainting_time_s, t1 - t0,
            result["overall_peak_gpu_gb"],
        )

    except Exception as exc:
        t_now = time.perf_counter()
        logger.error("[%d] ERROR: %s", idx, exc)
        (sample_dir / "error.txt").write_text(traceback.format_exc())
        result = {
            "index": idx,
            "inpainting_time_s": None,
            "total_time_s": round(t_now - t0, 2),
            "inpainting_peak_gpu_gb": None,
            "overall_peak_gpu_gb": None,
            "error": True,
            "error_message": f"{type(exc).__name__}: {exc}",
        }
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(tmp, ignore_errors=True)

    (sample_dir / "timing.json").write_text(json.dumps(result, indent=2))
    return result


def _summarize(values: list[float]) -> dict:
    a = np.array(values)
    return {
        "mean": round(float(a.mean()), 3),
        "std":  round(float(a.std()),  3),
        "min":  round(float(a.min()),  3),
        "max":  round(float(a.max()),  3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark vanilla FLUX.1 Fill timing on all real images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", type=str, default="outputs/timing_benchmark_vanilla")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--indices", type=int, nargs="*", default=None,
        help="Specific indices to process. Default: all 0-49.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=float, default=1024)
    parser.add_argument("--width", type=float, default=1024)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=30.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip samples where timing.json already exists.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config))

    indices = args.indices if args.indices is not None else list(range(N_REAL_IMAGES))

    print(f"Output dir : {output_dir}")
    print(f"Seed       : {args.seed}")
    print(f"Indices    : {len(indices)} samples ({indices[0]}-{indices[-1]})")
    print()

    r2 = R2Client()

    print("Loading vanilla FLUX.1 Fill pipeline ...")
    pipe = load_vanilla_pipeline()
    print("Pipeline loaded.\n")

    rows: list[dict] = []
    total = len(indices)
    for i, idx in enumerate(indices):
        timing_path = output_dir / f"sample_{idx}" / "timing.json"
        if args.skip_existing and timing_path.exists():
            existing = json.loads(timing_path.read_text())
            rows.append(existing)
            print(f"[{i + 1}/{total}] index {idx} — skipping (timing.json exists)")
            continue

        print(f"[{i + 1}/{total}] index {idx} ...")
        result = benchmark_sample(idx, r2, pipe, config, output_dir, args)
        rows.append(result)

    # Aggregate
    ok_rows = [r for r in rows if not r.get("error")]
    n_ok = len(ok_rows)
    n_err = len(rows) - n_ok
    print(f"\n{n_ok}/{len(rows)} samples succeeded ({n_err} errors)")

    if not ok_rows:
        logger.error("No successful samples.")
        return

    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    df.to_csv(output_dir / "timing_summary.csv", index=False)

    timing_fields = [
        "inpainting_time_s", "total_time_s",
        "inpainting_peak_gpu_gb", "overall_peak_gpu_gb",
    ]
    summary: dict = {"n_samples": n_ok, "n_errors": n_err}
    for field in timing_fields:
        vals = [r[field] for r in ok_rows if r.get(field) is not None]
        if vals:
            summary[field] = _summarize(vals)

    (output_dir / "timing_summary.json").write_text(json.dumps(summary, indent=2))

    col_w = 30
    print("\n" + "=" * 65)
    print(f"Vanilla FLUX.1 Fill Timing Benchmark  (N={n_ok} real images, seed={args.seed})")
    print(f"{'':>{col_w}}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}")
    print("-" * 65)
    for field in timing_fields:
        if field not in summary:
            continue
        s = summary[field]
        unit = "(GB)" if "gpu" in field else "(s) "
        print(f"{field + ' ' + unit:<{col_w}}{s['mean']:8.2f}{s['std']:8.2f}{s['min']:8.2f}{s['max']:8.2f}")
    print("=" * 65)
    print(f"\nSaved: {output_dir / 'timing_summary.csv'}")
    print(f"Saved: {output_dir / 'timing_summary.json'}")


if __name__ == "__main__":
    main()
