"""
Inpainting experiment script using QwenImageEditInpaintPipeline.

Downloads projection outputs from Cloudflare R2, runs QwenImageEditInpaintPipeline
for all seeds across multiple GPUs, and uploads results back to R2.

Output layout per image:
    <output_root>/<dataset>/<geom_subdir>/<model_slug>/<index>/
        seed_<seed>.png
        seed_<seed>_metadata.json   # prompt, inpainting_time_s

Missing projected images are logged to:
    <output_root>/worker_<gpu_id>_errors.txt

Examples
--------
Single GPU, single seed:

    python scripts/run_qwen_inpainting_experiment.py \\
        --dataset mirrorbench_v2 \\
        --geom-subdir estimated_geometry \\
        --output-root /tmp/qwen_inpaint_experiment \\
        --seeds 0

Multi-GPU, multiple seeds:

    python scripts/run_qwen_inpainting_experiment.py \\
        --dataset mirrorbench_v2 \\
        --geom-subdir estimated_geometry \\
        --output-root /tmp/qwen_inpaint_experiment \\
        --seeds 0 1 42 \\
        --num-gpus 2
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing
import time
import traceback
from pathlib import Path

import torch
from diffusers import QwenImageEditInpaintPipeline
from PIL import Image

from fill_my_mirror.storage import R2Client

MODEL_NAME = "Qwen/Qwen-Image-Edit"
DEFAULT_UPLOAD_EVERY = 15
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_NUM_INFERENCE_STEPS = 30

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------

def _load_pipeline(gpu_id: int) -> QwenImageEditInpaintPipeline:
    device = f"cuda:{gpu_id}"
    pipe = QwenImageEditInpaintPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    pipe.to(device)
    return pipe


def _worker(
    gpu_id: int,
    indices: list[int],
    args_dict: dict,
) -> None:
    dataset = args_dict["dataset"]
    geom_subdir = args_dict["geom_subdir"]
    output_root = Path(args_dict["output_root"])
    seeds = args_dict["seeds"]
    upload_every = args_dict["upload_every"]
    skip_existing = args_dict["skip_existing"]
    height = args_dict["height"]
    width = args_dict["width"]
    num_inference_steps = args_dict["num_inference_steps"]

    model_slug = MODEL_NAME.replace("/", "--")

    error_log = output_root / f"worker_{gpu_id}_errors.txt"
    output_root.mkdir(parents=True, exist_ok=True)

    def log_error(msg: str) -> None:
        with open(error_log, "a") as f:
            f.write(msg + "\n")
        print(f"[GPU {gpu_id}] ERROR: {msg}")

    r2 = R2Client()
    pipe = _load_pipeline(gpu_id)
    device = f"cuda:{gpu_id}"

    pending_upload: list[tuple[Path, str]] = []

    def flush():
        for local_dir, r2_prefix in pending_upload:
            r2.upload_dir(local_dir, r2_prefix)
        pending_upload.clear()

    total = len(indices) * len(seeds)
    done = 0

    for index in indices:
        import shutil
        import tempfile

        r2_proj_key = f"{dataset}/{geom_subdir}/{index}"

        proj_tmp = Path(tempfile.mkdtemp(prefix=f"proj_{index}_"))
        try:
            # Download required files
            missing = False
            for fname in ("original_image.png", "generative_refinement_mask.png"):
                try:
                    r2.download_file(f"{r2_proj_key}/{fname}", proj_tmp / fname)
                except Exception:
                    log_error(f"index {index}: missing {fname} in {r2_proj_key}")
                    missing = True
                    break

            if missing:
                shutil.rmtree(proj_tmp, ignore_errors=True)
                continue

            # Download prompt
            try:
                r2.download_file(f"{r2_proj_key}/prompt.json", proj_tmp / "prompt.json")
                prompt_data = json.loads((proj_tmp / "prompt.json").read_text())
            except Exception:
                prompt_data = {}

            prompt = prompt_data.get("prompt", "").strip()
            if not prompt:
                log_error(f"index {index}: missing or empty prompt, skipping")
                shutil.rmtree(proj_tmp, ignore_errors=True)
                continue

            original_image = Image.open(proj_tmp / "original_image.png").convert("RGB")
            mask_image = Image.open(proj_tmp / "generative_refinement_mask.png").convert("L")

            index_dir = output_root / dataset / geom_subdir / model_slug / str(index)
            index_dir.mkdir(parents=True, exist_ok=True)
            r2_out_prefix = f"{dataset}/{geom_subdir}/{model_slug}/{index}"

            for seed in seeds:
                done += 1
                label = f"[GPU {gpu_id}] [{done}/{total}] index {index} seed={seed}"

                output_path = index_dir / f"seed_{seed}.png"
                metadata_path = index_dir / f"seed_{seed}_metadata.json"

                if skip_existing and r2.key_exists(f"{r2_out_prefix}/seed_{seed}.png"):
                    print(f"{label} — skipping (already in R2)")
                    continue

                generator = torch.Generator(device=device).manual_seed(seed)

                t_start = time.perf_counter()
                try:
                    print(f"{label} — generating ...")
                    result = pipe(
                        image=original_image,
                        prompt=prompt,
                        negative_prompt=" ",
                        mask_image=mask_image,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                    )
                    result.images[0].save(output_path)
                    t_elapsed = time.perf_counter() - t_start

                    metadata = {
                        "prompt": prompt,
                        "inpainting_time_s": round(t_elapsed, 2),
                        "model": MODEL_NAME,
                        "height": height,
                        "width": width,
                        "num_inference_steps": num_inference_steps,
                        "seed": seed,
                    }
                    metadata_path.write_text(json.dumps(metadata, indent=2))
                    print(f"{label} — done ({t_elapsed:.1f}s), queued for upload")

                except Exception:
                    t_elapsed = time.perf_counter() - t_start
                    print(f"{label} — ERROR after {t_elapsed:.1f}s")
                    traceback.print_exc()

                gc.collect()
                torch.cuda.empty_cache()

            pending_upload.append((index_dir, r2_out_prefix))

            if len(pending_upload) >= upload_every:
                flush()

        finally:
            shutil.rmtree(proj_tmp, ignore_errors=True)

    flush()
    print(f"[GPU {gpu_id}] done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen inpainting experiment from R2 projection outputs")
    parser.add_argument("--dataset", required=True, choices=["blender", "real", "mirrorbench_v2"])
    parser.add_argument("--geom-subdir", default="gt_geometry", choices=["gt_geometry", "estimated_geometry"])
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--upload-every", type=int, default=DEFAULT_UPLOAD_EVERY)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images that have already been generated in R2")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found. Run with --num-gpus to override.")

    model_slug = MODEL_NAME.replace("/", "--")

    # Discover available indices from R2
    r2 = R2Client()
    prefix = f"{args.dataset}/{args.geom_subdir}/"
    keys = r2.list_keys(prefix)
    all_indices = sorted({int(k.split("/")[2]) for k in keys if k.split("/")[2].isdigit()})
    start = args.start_index
    end = args.end_index if args.end_index is not None else (max(all_indices) + 1 if all_indices else 0)
    indices = [i for i in all_indices if start <= i < end]

    print(f"Model    : {MODEL_NAME}")
    print(f"Dataset  : {args.dataset}/{args.geom_subdir}")
    print(f"Indices  : {len(indices)} ({start}–{end})")
    print(f"Seeds    : {args.seeds}")
    print(f"GPUs     : {num_gpus}")
    print(f"Steps    : {args.num_inference_steps}")
    print(f"Size     : {args.height}x{args.width}")
    print()

    args_dict = {
        "dataset": args.dataset,
        "geom_subdir": args.geom_subdir,
        "output_root": str(args.output_root),
        "seeds": args.seeds,
        "upload_every": args.upload_every,
        "skip_existing": args.skip_existing,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
    }

    if num_gpus == 1:
        _worker(0, indices, args_dict)
    else:
        chunk = (len(indices) + num_gpus - 1) // num_gpus
        chunks = [indices[i * chunk:(i + 1) * chunk] for i in range(num_gpus)]
        processes = []
        for gpu_id, chunk_indices in enumerate(chunks):
            if not chunk_indices:
                continue
            p = multiprocessing.Process(
                target=_worker,
                args=(gpu_id, chunk_indices, args_dict),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
