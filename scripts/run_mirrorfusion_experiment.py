"""
MirrorFusion inference script on MirrorBench-V2.

Reads HDF5 files from a local data root, runs StableDiffusionBrushNetPipeline
(MirrorFusion) for all seeds across multiple GPUs, and uploads results to R2.

Local data root layout (default: ~/data/mirrorbench_v2):
    test.csv
    <row["path"]>   (e.g. abo_v4/B/B07JY4H14B/0.hdf5)

Output layout per sample in R2:
    mirrorbench_v2/<model_slug>/<index>/
        seed_<seed>.png
        seed_<seed>_metadata.json   # prompt, inference_time_s, settings

Errors are logged to:
    <output_root>/worker_<gpu_id>_errors.txt

Examples
--------
Single GPU, single seed:

    python scripts/run_mirrorfusion_experiment.py \\
        --brushnet-path ~/weights/mirror_fusion_v2 \\
        --output-root /tmp/mirrorfusion_experiment \\
        --seeds 0

Multi-GPU, multiple seeds:

    python scripts/run_mirrorfusion_experiment.py \\
        --brushnet-path ~/weights/mirror_fusion_v2 \\
        --output-root /tmp/mirrorfusion_experiment \\
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

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from diffusers import (
    BrushNetModel,
    StableDiffusionBrushNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from PIL import Image

from fill_my_mirror.storage import R2Client

DATASET = "mirrorbench_v2"
DEFAULT_DATA_ROOT = Path.home() / "data" / "mirrorbench_v2"
BASE_MODEL_PATH = "runwayml/stable-diffusion-v1-5"
MIRROR_PROMPT = "A perfect plane mirror reflection of "
DEFAULT_UPLOAD_EVERY = 15
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_BRUSHNET_CONDITIONING_SCALE = 1.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HDF5 helpers (inlined from Reflecting-Reality/MirrorFusion/dataset/dataset.py
# to avoid a fragile sys.path dependency)
# ---------------------------------------------------------------------------

def _extract_data_from_hdf5(hdf5_data) -> dict:
    image = np.array(hdf5_data["colors"], dtype=np.uint8)
    mask = (np.array(hdf5_data["category_id_segmaps"], dtype=np.uint8) == 1).astype(np.uint8) * 255
    masked_image = image.copy()
    masked_image[mask == 255] = 0
    return {
        "image": image,
        "mask": mask,
        "masked_image": masked_image,
        "depth": np.array(hdf5_data["depth"]),
        "normals": np.array(hdf5_data["normals"]),
    }


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
    rows: list[dict],
    args_dict: dict,
) -> None:
    brushnet_path = args_dict["brushnet_path"]
    base_model_path = args_dict["base_model_path"]
    data_root = Path(args_dict["data_root"])
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

    total = len(rows) * len(seeds)
    done = 0

    for row in rows:
        index = row["index"]
        hdf5_path = data_root / row["path"]
        caption = row["caption"]

        if not hdf5_path.exists():
            log_error(f"index {index}: HDF5 not found at {hdf5_path}")
            continue

        try:
            with h5py.File(hdf5_path, "r") as f:
                data = _extract_data_from_hdf5(f)

            validation_image = Image.fromarray(data["masked_image"], mode="RGB").resize(
                (width, height), Image.LANCZOS
            )
            validation_mask = Image.fromarray(data["mask"]).convert("RGB").resize(
                (width, height), Image.NEAREST
            )

            depth_tensor = None
            if depth_conditioning_mode is not None:
                depth_tensor = _apply_transforms_depth(data["depth"], data["mask"], resolution=height)

            normal_image = None
            if normals_conditioning_mode is not None:
                normal_image = Image.fromarray(data["normals"], mode="RGB").resize(
                    (width, height), Image.LANCZOS
                )

            validation_prompt = MIRROR_PROMPT + caption

            index_dir = output_root / DATASET / model_slug / str(index)
            index_dir.mkdir(parents=True, exist_ok=True)
            r2_out_prefix = f"{DATASET}/{model_slug}/gt_geometry/{index}"

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
                        validation_prompt,
                        validation_image,
                        validation_mask,
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
                        "prompt": validation_prompt,
                        "uid": row["uid"],
                        "hdf5_path": row["path"],
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

        except Exception:
            log_error(f"index {index}: {traceback.format_exc()}")

    flush()
    print(f"[GPU {gpu_id}] done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MirrorFusion inference on MirrorBench-V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--brushnet-path", required=True, type=str,
                        help="Weights dir containing brushnet/ and unet/ subdirs")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"Local MirrorBench-V2 root (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--base-model-path", default=BASE_MODEL_PATH)
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
    parser.add_argument("--brushnet-conditioning-scale", type=float, default=DEFAULT_BRUSHNET_CONDITIONING_SCALE)
    parser.add_argument("--depth-conditioning-mode", default="concat",
                        choices=["concat", "latents"],
                        help="How depth is injected into the model (default: concat)")
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable depth conditioning entirely")
    parser.add_argument("--normals-conditioning-mode", default=None,
                        choices=["concat", "latents"],
                        help="Enable normals conditioning (default: disabled)")
    parser.add_argument("--caption-column", default="auto_caption",
                        choices=["caption", "auto_caption"])
    parser.add_argument("--csv", default="test.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found. Use --num-gpus to override.")

    depth_conditioning_mode = None if args.no_depth else args.depth_conditioning_mode

    # Build slug from key settings so outputs are self-documenting
    parts = ["mirrorfusion"]
    parts.append(f"depth_{depth_conditioning_mode}" if depth_conditioning_mode else "nodepth")
    if args.normals_conditioning_mode:
        parts.append(f"normals_{args.normals_conditioning_mode}")
    model_slug = "_".join(parts)

    data_root = args.data_root.expanduser()
    df = pd.read_csv(data_root / args.csv)

    all_rows = [
        {
            "index": i,
            "path": str(row["path"]),
            "uid": str(row["uid"]),
            "caption": str(row[args.caption_column]),
        }
        for i, row in df.iterrows()
    ]
    start = args.start_index
    end = args.end_index if args.end_index is not None else len(all_rows)
    rows = [r for r in all_rows if start <= r["index"] < end]

    print(f"Data root : {data_root}")
    print(f"Model     : {Path(args.brushnet_path).name}")
    print(f"Slug      : {model_slug}")
    print(f"Dataset   : {DATASET}/{args.csv}")
    print(f"Rows      : {len(rows)} ({start}–{end-1})")
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
        "data_root": str(data_root),
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
        _worker(0, rows, args_dict)
    else:
        chunk_size = (len(rows) + num_gpus - 1) // num_gpus
        chunks = [rows[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]
        processes = []
        for gpu_id, chunk_rows in enumerate(chunks):
            if not chunk_rows:
                continue
            p = multiprocessing.Process(target=_worker, args=(gpu_id, chunk_rows, args_dict))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
