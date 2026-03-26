from __future__ import annotations

from pathlib import Path

import torch
from diffusers.utils import load_image

from .pipeline import DualMaskInterpolatedFluxFillPipeline


def run_dual_mask_inpainting(
    prompt: str,
    projected_image_path: str | Path,
    geometry_constraint_mask_path: str | Path,
    generative_refinement_mask_path: str | Path,
    output_path: str | Path,
    model_name: str = "black-forest-labs/FLUX.1-Fill-dev",
    prompt_2: str | None = None,
    strength: float = 1.0,
    num_inference_steps: int = 30,
    guidance_scale: float = 30.0,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    seed: int = 0,
    n: float = 6.0,
    t_prime: float = 750.0,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    projected_image_path = Path(projected_image_path)
    geometry_constraint_mask_path = Path(geometry_constraint_mask_path)
    generative_refinement_mask_path = Path(generative_refinement_mask_path)
    output_path = Path(output_path)

    image = load_image(str(projected_image_path)).convert("RGB")
    geometry_constraint_mask = load_image(str(geometry_constraint_mask_path)).convert("L")
    generative_refinement_mask = load_image(str(generative_refinement_mask_path)).convert("L")

    if image.size != geometry_constraint_mask.size:
        raise ValueError(
            f"Projected image and geometry constraint mask must have the same size, "
            f"got image={image.size}, geometry_constraint_mask={geometry_constraint_mask.size}"
        )

    if image.size != generative_refinement_mask.size:
        generative_refinement_mask = generative_refinement_mask.resize(image.size)

    pipe = DualMaskInterpolatedFluxFillPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).to(device)

    generator = torch.Generator("cpu").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        prompt_2=prompt_2,
        image=image,
        geometry_constraint_mask=geometry_constraint_mask,
        generative_refinement_mask=generative_refinement_mask,
        height=image.height,
        width=image.width,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        generator=generator,
        n=n,
        t_prime=t_prime,
    )

    final_image = result.images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(output_path)

    return final_image