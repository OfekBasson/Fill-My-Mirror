"""
Local inpainting script using FluxFillPipeline or QwenImageEditInpaintPipeline.

Examples
--------
FLUX Fill:

    python scripts/inpaint_local.py \\
        --image /path/to/image.png \\
        --mask /path/to/mask.png \\
        --prompt "a mirror reflecting a living room" \\
        --output /tmp/result.png

Qwen:

    python scripts/inpaint_local.py \\
        --model qwen \\
        --image /path/to/image.png \\
        --mask /path/to/mask.png \\
        --prompt "a mirror reflecting a living room" \\
        --output /tmp/result.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

FLUX_MODEL = "black-forest-labs/FLUX.1-Fill-dev"
QWEN_MODEL = "Qwen/Qwen-Image-Edit-2511"

DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 30.0
DEFAULT_MAX_SEQUENCE_LENGTH = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inpaint a local image using FLUX Fill or Qwen")
    parser.add_argument("--model", choices=["flux", "qwen"], default="flux")
    parser.add_argument("--image", required=True, type=Path, help="Path to input image")
    parser.add_argument("--mask", required=True, type=Path, help="Path to mask image (white = inpaint region)")
    parser.add_argument("--prompt", required=True, help="Inpainting prompt")
    parser.add_argument("--output", required=True, type=Path, help="Path to save output image", default="outputs/local")
    parser.add_argument("--seed", type=int, default=512)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    # FLUX-only
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    # Qwen-only
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")

    if args.height == DEFAULT_HEIGHT and args.width == DEFAULT_WIDTH:
        args.width, args.height = image.size

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model  : {args.model}")
    print(f"Device : {device}")
    print(f"Prompt : {args.prompt}")
    print(f"Image  : {args.image}")
    print(f"Mask   : {args.mask}")
    print(f"Output : {args.output}")
    print()

    t_start = time.perf_counter()

    if args.model == "flux":
        from diffusers import FluxFillPipeline

        pipe = FluxFillPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16).to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed)

        result = pipe(
            prompt=args.prompt,
            image=image,
            mask_image=mask,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            max_sequence_length=args.max_sequence_length,
            generator=generator,
        )

        result.images[0].save(args.output)
    else:
        from diffusers import QwenImageEditInpaintPipeline

        pipe = QwenImageEditInpaintPipeline.from_pretrained(QWEN_MODEL, torch_dtype=torch.bfloat16).to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed)

        result = pipe(
            image=image,
            prompt=args.prompt,
            negative_prompt=" ",
            mask_image=mask,
            strength=args.strength,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
        result.images[0].save(args.output)

    elapsed = time.perf_counter() - t_start
    print(f"Done in {elapsed:.1f}s — saved to {args.output}")


if __name__ == "__main__":
    main()
