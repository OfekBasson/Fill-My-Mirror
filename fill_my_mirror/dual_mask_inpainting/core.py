from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from diffusers.utils import load_image
from diffusers.utils import logging

from .pipeline_flux1 import DualMaskInterpolatedFluxFillPipeline
from .pipeline_qwen_inpaint import DualMaskInterpolatedQwenInpaintPipeline


logger = logging.get_logger(__name__)

MODEL_REGISTRY: dict[str, type] = {
    "black-forest-labs/FLUX.1-Fill-dev": DualMaskInterpolatedFluxFillPipeline,
    "Qwen/Qwen-Image-Edit-2511": DualMaskInterpolatedQwenInpaintPipeline,
}

# General-purpose edit models need the prompt to explicitly describe the mirror-filling
# task; native inpainting models receive the content prompt directly.
_EDIT_MODEL_PROMPT_TEMPLATE = (
    "Fill in the mirror corresponding to the mask, taking into account the provided "
    "geometry information to maintain geometric consistency. "
    "The prompt describing the image is: {prompt}"
)
_EDIT_MODELS = {
    "Qwen/Qwen-Image-Edit-2511",
}


def _apply_prompt_template(prompt: str, model_name: str) -> str:
    if model_name in _EDIT_MODELS:
        return _EDIT_MODEL_PROMPT_TEMPLATE.format(prompt=prompt)
    return prompt


PIPELINE_SUPPORTED_KWARGS: dict[type, set[str]] = {
    DualMaskInterpolatedFluxFillPipeline: {
        "prompt",
        "prompt_2",
        "image",
        "geometry_constraint_mask_image",
        "generative_refinement_mask_image",
        "height",
        "width",
        "strength",
        "num_inference_steps",
        "guidance_scale",
        "num_images_per_prompt",
        "max_sequence_length",
        "generator",
        "n",
        "t_prime",
    },
    DualMaskInterpolatedQwenInpaintPipeline: {
        "prompt",
        "negative_prompt",
        "image",
        "geometry_constraint_mask_image",
        "generative_refinement_mask_image",
        "height",
        "width",
        "strength",
        "num_inference_steps",
        "guidance_scale",
        "true_cfg_scale",
        "num_images_per_prompt",
        "max_sequence_length",
        "generator",
        "n",
        "t_prime",
    },
}


def load_inpainting_pipeline(
    model_name: str = "black-forest-labs/FLUX.1-Fill-dev",
    pipeline_class: type | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Any:
    """Load and return the inpainting pipeline.

    Call this once and pass the result to ``run_dual_mask_inpainting`` to avoid
    reloading the model on every iteration.
    """

    if pipeline_class is None:
        pipeline_class = MODEL_REGISTRY.get(model_name)
        if pipeline_class is None:
            supported = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unsupported inpainting model `{model_name}`. Supported models: {supported}")

    pipe = pipeline_class.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def _filter_pipeline_kwargs(pipe: Any, model_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    supported_kwargs = PIPELINE_SUPPORTED_KWARGS.get(type(pipe))
    if supported_kwargs is None:
        supported_kwargs = set(kwargs)
        logger.warning(
            "No explicit supported-argument list is registered for pipeline class %s; passing all arguments.",
            type(pipe).__name__,
        )

    filtered = {key: value for key, value in kwargs.items() if key in supported_kwargs}
    unsupported = sorted(
        key for key in kwargs if key not in supported_kwargs and kwargs[key] is not None and kwargs[key] is not False
    )
    if unsupported:
        logger.warning(
            "Model `%s` with pipeline `%s` does not support arguments %s; ignoring them.",
            model_name,
            type(pipe).__name__,
            ", ".join(unsupported),
        )
    return filtered


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
    height: int = 1024,
    width: int = 1024,
    n: float = 6.0,
    t_prime: float = 750.0,
    torch_dtype: torch.dtype = torch.bfloat16,
    pipe: Any | None = None,
):
    projected_image_path = Path(projected_image_path)
    geometry_constraint_mask_path = Path(geometry_constraint_mask_path)
    generative_refinement_mask_path = Path(generative_refinement_mask_path)
    output_path = Path(output_path)

    image = load_image(str("/home/ofek_basson/Fill-My-Mirror/data/real_images/images/0.png")).convert("RGB")
    # image = load_image(str(projected_image_path)).convert("RGB")
    geometry_constraint_mask = load_image(str(geometry_constraint_mask_path)).convert("L")
    generative_refinement_mask = load_image(str(generative_refinement_mask_path)).convert("L")

    if image.size != geometry_constraint_mask.size:
        raise ValueError(
            f"Projected image and geometry constraint mask must have the same size, "
            f"got image={image.size}, geometry_constraint_mask={geometry_constraint_mask.size}"
        )

    if image.size != generative_refinement_mask.size:
        generative_refinement_mask = generative_refinement_mask.resize(image.size)

    if pipe is None:
        pipe = load_inpainting_pipeline(
            model_name=model_name,
            pipeline_class=MODEL_REGISTRY.get(model_name),
            torch_dtype=torch_dtype,
        )

    generator = torch.Generator("cuda").manual_seed(seed)
    prompt = _apply_prompt_template(prompt, model_name)

    pipeline_kwargs = _filter_pipeline_kwargs(
        pipe,
        model_name,
        {
            "prompt": prompt,
            "prompt_2": prompt_2,
            "image": image,
            # "geometry_constraint_mask_image": geometry_constraint_mask,
            "geometry_constraint_mask_image": generative_refinement_mask,
            "generative_refinement_mask_image": generative_refinement_mask,
            "height": height,
            "width": width,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "max_sequence_length": max_sequence_length,
            "generator": generator,
            "n": n,
            "t_prime": t_prime,
        },
    )
    print(f'running pipeline with kwargs: {pipeline_kwargs}')
    result = pipe(
        **pipeline_kwargs,
    )

    final_image = result.images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(output_path)

    return final_image
