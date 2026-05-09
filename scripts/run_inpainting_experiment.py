"""
Inpainting experiment script for Fill My Mirror.

Downloads projection outputs from Cloudflare R2, runs run_dual_mask_inpainting
for all combinations of (n, t_prime) × seeds across multiple GPUs, and uploads
results back to R2.

Output layout per image:
    <output_root>/<dataset>/<geom_subdir>/<model_slug>/n_<n>_t_<t>/<index>/
        seed_<seed>.png
        seed_<seed>_metadata.json   # prompt, inpainting_time_s

    <output_root>/<dataset>/<geom_subdir>/<model_slug>/n_<n>_t_<t>/
        params.json     # all hyperparameters for this (n, t) combo

Missing projected images are logged to:
    <output_root>/worker_<gpu_id>_errors.txt

Examples
--------
Single GPU, single (n, t) combo, single seed:

    python scripts/run_inpainting_experiment.py \\
        --dataset mirrorbench_v2 \\
        --geom-subdir estimated_geometry \\
        --output-root /tmp/inpaint_experiment \\
        --seeds 0 \\
        --ns 6.0 \\
        --ts 750.0

Multi-GPU, multiple (n, t) combos, multiple seeds:

    python scripts/run_inpainting_experiment.py \\
        --dataset mirrorbench_v2 \\
        --geom-subdir estimated_geometry \\
        --output-root /tmp/inpaint_experiment \\
        --seeds 0 1 42 \\
        --ns 4.0 6.0 8.0 \\
        --ts 500.0 750.0 \\
        --num-gpus 2
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import multiprocessing
import os
import time
import traceback
from pathlib import Path

import torch
import yaml

from fill_my_mirror.dual_mask_inpainting import load_inpainting_pipeline, run_dual_mask_inpainting
from fill_my_mirror.storage import R2Client

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
DEFAULT_UPLOAD_EVERY = 15

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------

def _worker(
    gpu_id: int,
    indices: list[int],
    args_dict: dict,
    base_params: dict,
    nt_combos: list[tuple[float, float]],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model_name = args_dict["model_name"]
    dataset = args_dict["dataset"]
    geom_subdir = args_dict["geom_subdir"]
    output_root = Path(args_dict["output_root"])
    seeds = args_dict["seeds"]
    upload_every = args_dict["upload_every"]
    prompt_2 = args_dict.get("prompt_2")

    error_log = output_root / f"worker_{gpu_id}_errors.txt"
    output_root.mkdir(parents=True, exist_ok=True)

    def log_error(msg: str) -> None:
        with open(error_log, "a") as f:
            f.write(msg + "\n")
        print(f"[GPU {gpu_id}] ERROR: {msg}")

    r2 = R2Client()
    pipe = load_inpainting_pipeline(model_name=model_name)

    model_slug = model_name.replace("/", "--")

    pending_upload: list[tuple[Path, str]] = []  # (local_dir, r2_prefix)

    def flush():
        for local_dir, r2_prefix in pending_upload:
            r2.upload_dir(local_dir, r2_prefix)
        pending_upload.clear()

    total = len(indices) * len(nt_combos) * len(seeds)
    done = 0

    for index in indices:
        import tempfile, shutil
        r2_proj_key = f"{dataset}/{geom_subdir}/{index}"

        proj_tmp = Path(tempfile.mkdtemp(prefix=f"proj_{index}_"))
        try:
            # Download projection outputs
            missing = False
            for fname in ("projected_image.png", "geometry_constraint_mask.png",
                          "generative_refinement_mask.png", "original_image.png", "gt_image.png"):
                try:
                    r2.download_file(f"{r2_proj_key}/{fname}", proj_tmp / fname)
                except Exception:
                    if fname == "projected_image.png":
                        missing = True
                        break

            if missing:
                log_error(f"index {index}: missing projected_image.png in {r2_proj_key}")
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

            # Check if projection failed — if error.txt exists, skip geometry guidance
            has_error = r2.key_exists(f"{r2_proj_key}/error.txt")
            use_dual_mask = not has_error
            geometry_constraint_mask_path = (
                proj_tmp / "generative_refinement_mask.png" if has_error
                else proj_tmp / "geometry_constraint_mask.png"
            )

            for n, t_prime in nt_combos:
                nt_slug = f"n_{n}_t_{t_prime}"
                params = {**base_params, "n": n, "t_prime": t_prime}

                nt_dir = output_root / dataset / geom_subdir / model_slug / nt_slug
                nt_dir.mkdir(parents=True, exist_ok=True)
                (nt_dir / "params.json").write_text(json.dumps({"model": model_name, **params}, indent=2))

                index_dir = nt_dir / str(index)
                index_dir.mkdir(parents=True, exist_ok=True)
                r2_out_prefix = f"{dataset}/{geom_subdir}/{model_slug}/{nt_slug}/{index}"

                for seed in seeds:
                    done += 1
                    label = f"[GPU {gpu_id}] [{done}/{total}] index {index} n={n} t={t_prime} seed={seed}"

                    output_path = index_dir / f"seed_{seed}.png"
                    metadata_path = index_dir / f"seed_{seed}_metadata.json"

                    if args_dict["skip_existing"] and r2.key_exists(f"{r2_out_prefix}/seed_{seed}.png"):
                        print(f"{label} — skipping (already in R2)")
                        continue

                    t_start = time.perf_counter()
                    try:
                        print(f"{label} — generating (use_dual_mask={use_dual_mask}) ...")
                        run_dual_mask_inpainting(
                            prompt=prompt,
                            projected_image_path=proj_tmp / "projected_image.png",
                            geometry_constraint_mask_path=geometry_constraint_mask_path,
                            generative_refinement_mask_path=proj_tmp / "generative_refinement_mask.png",
                            output_path=output_path,
                            original_image_path=proj_tmp / "original_image.png",
                            model_name=model_name,
                            prompt_2=prompt_2,
                            seed=seed,
                            use_dual_mask=use_dual_mask,
                            pipe=pipe,
                            **params,
                        )
                        t_elapsed = time.perf_counter() - t_start

                        metadata = {
                            "prompt": prompt,
                            "inpainting_time_s": round(t_elapsed, 2),
                            "use_dual_mask": use_dual_mask,
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
    parser = argparse.ArgumentParser(description="Run inpainting experiment from R2 projection outputs")
    parser.add_argument("--dataset", required=True, choices=["blender", "real", "mirrorbench_v2"])
    parser.add_argument("--geom-subdir", default="gt_geometry", choices=["gt_geometry", "estimated_geometry"])
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model-name", default=None, help="Inpainting model (overrides config)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--ns", nargs="+", type=float, default=[6.0], help="Values of n to sweep")
    parser.add_argument("--ts", nargs="+", type=float, default=[750.0], help="Values of t_prime to sweep")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--upload-every", type=int, default=DEFAULT_UPLOAD_EVERY)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images that have already been generated locally")
    # Fixed inpainting hyperparameters
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=30.0)
    parser.add_argument("--num-images-per-prompt", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--prompt-2", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = args.model_name or config["inpainting_model_name"]
    num_gpus = args.num_gpus or torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found. Run with --num-gpus to override.")

    base_params = {
        "strength": args.strength,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "num_images_per_prompt": args.num_images_per_prompt,
        "max_sequence_length": args.max_sequence_length,
        "height": args.height,
        "width": args.width,
    }
    nt_combos = list(itertools.product(args.ns, args.ts))

    # Discover available indices from R2
    r2 = R2Client()
    prefix = f"{args.dataset}/{args.geom_subdir}/"
    keys = r2.list_keys(prefix)
    all_indices = sorted({int(k.split("/")[2]) for k in keys if k.split("/")[2].isdigit()})
    start = args.start_index
    end = args.end_index if args.end_index is not None else (max(all_indices) + 1 if all_indices else 0)
    indices = [i for i in all_indices if start <= i < end]

    model_slug = model_name.replace("/", "--")

    print(f"Model    : {model_name}")
    print(f"Dataset  : {args.dataset}/{args.geom_subdir}")
    print(f"Indices  : {len(indices)} ({start}–{end})")
    print(f"Seeds    : {args.seeds}")
    print(f"(n, t)   : {nt_combos}")
    print(f"GPUs     : {num_gpus}")
    print()

    # Write params.json for each (n, t) combo upfront
    for n, t_prime in nt_combos:
        nt_slug = f"n_{n}_t_{t_prime}"
        nt_dir = args.output_root / args.dataset / args.geom_subdir / model_slug / nt_slug
        nt_dir.mkdir(parents=True, exist_ok=True)
        params = {**base_params, "n": n, "t_prime": t_prime}
        (nt_dir / "params.json").write_text(json.dumps({"model": model_name, **params}, indent=2))

    args_dict = {
        "model_name": model_name,
        "dataset": args.dataset,
        "geom_subdir": args.geom_subdir,
        "output_root": str(args.output_root),
        "seeds": args.seeds,
        "upload_every": args.upload_every,
        "skip_existing": args.skip_existing,
        "prompt_2": args.prompt_2,
    }

    if num_gpus == 1:
        _worker(0, indices, args_dict, base_params, nt_combos)
    else:
        chunk = (len(indices) + num_gpus - 1) // num_gpus
        chunks = [indices[i * chunk:(i + 1) * chunk] for i in range(num_gpus)]
        processes = []
        for gpu_id, chunk_indices in enumerate(chunks):
            if not chunk_indices:
                continue
            p = multiprocessing.Process(
                target=_worker,
                args=(gpu_id, chunk_indices, args_dict, base_params, nt_combos),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
