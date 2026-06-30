"""
MirrorFusion inference script reading inputs from Cloudflare R2.

Downloads projection outputs from R2 (original_image.png,
generative_refinement_mask.png, prompt.json, depth.npy), runs
StableDiffusionBrushNetPipeline (MirrorFusion) for all seeds across
multiple GPUs, and uploads results back to R2.

Expected R2 input layout per index:
    <dataset>/<geom_subdir>/<index>/
        original_image.png
        generative_refinement_mask.png
        prompt.json
        depth.npy          (required unless --no-depth)

Output layout per sample in R2:
    <dataset>/<geom_subdir>/<model_slug>/<index>/
        seed_<seed>.png
        seed_<seed>_metadata.json

Errors are logged to:
    <output_root>/worker_<gpu_id>_errors.txt

Examples
--------
Single GPU, single seed:

    python scripts/run_mirrorfusion_r2_experiment.py \\
        --brushnet-path ~/weights/mirror_fusion_v2 \\
        --dataset real \\
        --geom-subdir estimated_geometry \\
        --output-root /tmp/mirrorfusion_r2_experiment \\
        --seeds 0

Multi-GPU, multiple seeds:

    python scripts/run_mirrorfusion_r2_experiment.py \\
        --brushnet-path ~/weights/mirror_fusion_v2 \\
        --dataset real \\
        --geom-subdir estimated_geometry \\
        --output-root /tmp/mirrorfusion_r2_experiment \\
        --seeds 0 1 42 \\
        --num-gpus 2
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# BrushNet classes live in the custom diffusers fork bundled with Reflecting-Reality.
_MIRRORFUSION_SRC = Path.home() / "MirrorVerse/Reflecting-Reality/MirrorFusion/src"
sys.path.insert(0, str(_MIRRORFUSION_SRC))

from diffusers import (  # noqa: E402
    BrushNetModel,
    StableDiffusionBrushNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

from fill_my_mirror.storage import R2Client

BASE_MODEL_PATH = "runwayml/stable-diffusion-v1-5"
DEFAULT_UPLOAD_EVERY = 15
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_BRUSHNET_CONDITIONING_SCALE = 1.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Depth helpers (mirrored from run_mirrorfusion_experiment.py)
# ---------------------------------------------------------------------------

def _apply_transforms_depth(
    depth_map: np.ndarray,
    mask: np.ndarray,
    resolution: int,
    delta: float = 0.5,
) -> torch.Tensor:
    depth_map = np.copy(depth_map)
    mask_2d = mask[:, :, 0] if mask.ndim == 3 else mask
    bool_mask = mask_2d > 0
    max_scene_depth = np.max(depth_map[bool_mask]) + delta
    clipped = np.clip(depth_map, 0, max_scene_depth)
    normalized = 2.0 * (clipped / max_scene_depth) - 1.0
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
    return T.Compose([
        T.Resize(resolution, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop((resolution, resolution)),
    ])(tensor)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def _load_pipeline(
    brushnet_path: str,
    base_model_path: str,
    depth_conditioning_mode: str | None,
    normals_conditioning_mode: str | None,
    gpu_id: int,
) -> StableDiffusionBrushNetPipeline:
    device = f"cuda:{gpu_id}"
    dtype = torch.float16

    subfolder = "brushnet" if Path(brushnet_path, "brushnet").is_dir() else ""
    brushnet = BrushNetModel.from_pretrained(brushnet_path, subfolder=subfolder, torch_dtype=dtype)

    unet = None
    if Path(brushnet_path, "unet").is_dir():
        unet = UNet2DConditionModel.from_pretrained(brushnet_path, subfolder="unet", torch_dtype=dtype)

    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        base_model_path,
        brushnet=brushnet,
        unet=unet,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        safety_checker=None,
        depth_conditioning_mode=depth_conditioning_mode,
        normals_conditioning_mode=normals_conditioning_mode,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    return pipe


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------

def _worker(
    gpu_id: int,
    indices: list[int],
    args_dict: dict,
) -> None:
    brushnet_path = args_dict["brushnet_path"]
    base_model_path = args_dict["base_model_path"]
    dataset = args_dict["dataset"]
    geom_subdir = args_dict["geom_subdir"]
    output_root = Path(args_dict["output_root"])
    seeds = args_dict["seeds"]
    upload_every = args_dict["upload_every"]
    skip_existing = args_dict["skip_existing"]
    height = args_dict["height"]
    width = args_dict["width"]
    num_inference_steps = args_dict["num_inference_steps"]
    guidance_scale = args_dict["guidance_scale"]
    brushnet_conditioning_scale = args_dict["brushnet_conditioning_scale"]
    depth_conditioning_mode = args_dict["depth_conditioning_mode"]
    normals_conditioning_mode = args_dict["normals_conditioning_mode"]
    model_slug = args_dict["model_slug"]

    output_root.mkdir(parents=True, exist_ok=True)
    error_log = output_root / f"worker_{gpu_id}_errors.txt"
    device = f"cuda:{gpu_id}"

    def log_error(msg: str) -> None:
        with open(error_log, "a") as f:
            f.write(msg + "\n")
        print(f"[GPU {gpu_id}] ERROR: {msg}")

    r2 = R2Client()
    pipe = _load_pipeline(
        brushnet_path, base_model_path,
        depth_conditioning_mode, normals_conditioning_mode, gpu_id,
    )

    pending_upload: list[tuple[Path, str]] = []

    def flush():
        for local_dir, r2_prefix in pending_upload:
            r2.upload_dir(local_dir, r2_prefix)
        pending_upload.clear()

    total = len(indices) * len(seeds)
    done = 0

    for index in indices:
        r2_in_prefix = f"{dataset}/{geom_subdir}/{index}"
        proj_tmp = Path(tempfile.mkdtemp(prefix=f"proj_{index}_"))

        try:
            # Download required image and mask
            missing = False
            for fname in ("original_image.png", "generative_refinement_mask.png"):
                try:
                    r2.download_file(f"{r2_in_prefix}/{fname}", proj_tmp / fname)
                except Exception:
                    log_error(f"index {index}: missing {fname} in {r2_in_prefix}")
                    missing = True
                    break
            if missing:
                shutil.rmtree(proj_tmp, ignore_errors=True)
                continue

            # Download prompt
            try:
                r2.download_file(f"{r2_in_prefix}/prompt.json", proj_tmp / "prompt.json")
                prompt_data = json.loads((proj_tmp / "prompt.json").read_text())
            except Exception:
                prompt_data = {}

            prompt = prompt_data.get("prompt", "").strip()
            if not prompt:
                log_error(f"index {index}: missing or empty prompt, skipping")
                shutil.rmtree(proj_tmp, ignore_errors=True)
                continue

            original_image = Image.open(proj_tmp / "original_image.png").convert("RGB").resize(
                (width, height), Image.LANCZOS
            )
            mask_image = Image.open(proj_tmp / "generative_refinement_mask.png").convert("RGB").resize(
                (width, height), Image.NEAREST
            )

            # Depth conditioning
            depth_tensor = None
            if depth_conditioning_mode is not None:
                try:
                    r2.download_file(f"{r2_in_prefix}/depth.npy", proj_tmp / "depth.npy")
                    depth_arr = np.load(proj_tmp / "depth.npy")
                    mask_arr = np.array(
                        Image.open(proj_tmp / "generative_refinement_mask.png").convert("L")
                    )
                    depth_tensor = _apply_transforms_depth(depth_arr, mask_arr, resolution=height)
                except Exception:
                    log_error(f"index {index}: failed to load depth, skipping depth conditioning\n{traceback.format_exc()}")

            # Normals conditioning (optional)
            normal_image = None
            if normals_conditioning_mode is not None:
                try:
                    r2.download_file(f"{r2_in_prefix}/normals.png", proj_tmp / "normals.png")
                    normal_image = Image.open(proj_tmp / "normals.png").convert("RGB").resize(
                        (width, height), Image.LANCZOS
                    )
                except Exception:
                    log_error(f"index {index}: failed to load normals, skipping normals conditioning\n{traceback.format_exc()}")

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
                        prompt,
                        original_image,
                        mask_image,
                        depth=depth_tensor,
                        normals=normal_image,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        brushnet_conditioning_scale=brushnet_conditioning_scale,
                    )
                    result.images[0].save(output_path)
                    t_elapsed = time.perf_counter() - t_start

                    metadata_path.write_text(json.dumps({
                        "prompt": prompt,
                        "inference_time_s": round(t_elapsed, 2),
                        "model_slug": model_slug,
                        "height": height,
                        "width": width,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "brushnet_conditioning_scale": brushnet_conditioning_scale,
                        "depth_conditioning_mode": depth_conditioning_mode,
                        "normals_conditioning_mode": normals_conditioning_mode,
                        "seed": seed,
                    }, indent=2))
                    print(f"{label} — done ({t_elapsed:.1f}s), queued for upload")

                except Exception:
                    log_error(f"index {index} seed {seed}: {traceback.format_exc()}")

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
    parser = argparse.ArgumentParser(
        description="Run MirrorFusion inference on R2 projection outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--brushnet-path", required=True, type=str,
                        help="Weights dir containing brushnet/ and unet/ subdirs")
    parser.add_argument("--base-model-path", default=BASE_MODEL_PATH)
    parser.add_argument("--dataset", required=True, choices=["blender", "real", "mirrorbench_v2"])
    parser.add_argument("--geom-subdir", default="estimated_geometry",
                        choices=["gt_geometry", "estimated_geometry"])
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--upload-every", type=int, default=DEFAULT_UPLOAD_EVERY)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip samples already present in R2")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--brushnet-conditioning-scale", type=float,
                        default=DEFAULT_BRUSHNET_CONDITIONING_SCALE)
    parser.add_argument("--depth-conditioning-mode", default="concat",
                        choices=["concat", "latents"],
                        help="How depth is injected into the model (default: concat)")
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable depth conditioning entirely")
    parser.add_argument("--normals-conditioning-mode", default=None,
                        choices=["concat", "latents"],
                        help="Enable normals conditioning (default: disabled)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found. Use --num-gpus to override.")

    depth_conditioning_mode = None if args.no_depth else args.depth_conditioning_mode

    parts = ["mirrorfusion"]
    parts.append(f"depth_{depth_conditioning_mode}" if depth_conditioning_mode else "nodepth")
    if args.normals_conditioning_mode:
        parts.append(f"normals_{args.normals_conditioning_mode}")
    model_slug = "_".join(parts)

    # Discover available indices from R2
    r2 = R2Client()
    prefix = f"{args.dataset}/{args.geom_subdir}/"
    keys = r2.list_keys(prefix)
    all_indices = sorted({int(k.split("/")[2]) for k in keys if k.split("/")[2].isdigit()})
    start = args.start_index
    end = args.end_index if args.end_index is not None else (max(all_indices) + 1 if all_indices else 0)
    indices = [i for i in all_indices if start <= i < end]

    print(f"Model     : {Path(args.brushnet_path).name}")
    print(f"Slug      : {model_slug}")
    print(f"Dataset   : {args.dataset}/{args.geom_subdir}")
    print(f"Indices   : {len(indices)} ({start}–{end - 1})")
    print(f"Seeds     : {args.seeds}")
    print(f"GPUs      : {num_gpus}")
    print(f"Steps     : {args.num_inference_steps}")
    print(f"CFG       : {args.guidance_scale}")
    print(f"Size      : {args.height}x{args.width}")
    print(f"Depth     : {depth_conditioning_mode}")
    print(f"Normals   : {args.normals_conditioning_mode}")
    print()

    args_dict = {
        "brushnet_path": str(args.brushnet_path),
        "base_model_path": args.base_model_path,
        "dataset": args.dataset,
        "geom_subdir": args.geom_subdir,
        "output_root": str(args.output_root),
        "seeds": args.seeds,
        "upload_every": args.upload_every,
        "skip_existing": args.skip_existing,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "brushnet_conditioning_scale": args.brushnet_conditioning_scale,
        "depth_conditioning_mode": depth_conditioning_mode,
        "normals_conditioning_mode": args.normals_conditioning_mode,
        "model_slug": model_slug,
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
            p = multiprocessing.Process(target=_worker, args=(gpu_id, chunk_indices, args_dict))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
