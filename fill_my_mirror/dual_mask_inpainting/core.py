from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from diffusers.utils import load_image
from diffusers.utils import logging

from .pipeline_flux1 import DualMaskInterpolatedFluxFillPipeline
from .pipeline_flux2klein import DualMaskFlux2KleinInpaintPipeline


logger = logging.get_logger(__name__)

MODEL_REGISTRY: dict[str, type] = {
    "black-forest-labs/FLUX.1-Fill-dev": DualMaskInterpolatedFluxFillPipeline,
    "black-forest-labs/FLUX.2-klein-base-9B": DualMaskFlux2KleinInpaintPipeline,
}

_DEFAULT_EDIT_MODEL_PROMPT_TEMPLATE = (
    "Fill in the mirror which corresponds to the mask. The prompt describing the image is: {prompt}"
)
_DEFAULT_EDIT_MODELS: frozenset[str] = frozenset()


def _apply_prompt_template(
    prompt: str,
    model_name: str,
    edit_model_prompt_template: str = _DEFAULT_EDIT_MODEL_PROMPT_TEMPLATE,
    edit_models: frozenset[str] | set[str] = _DEFAULT_EDIT_MODELS,
) -> str:
    if model_name in edit_models:
        return edit_model_prompt_template.format(prompt=prompt)
    return prompt


PIPELINE_SUPPORTED_KWARGS: dict[type, set[str]] = {
    DualMaskFlux2KleinInpaintPipeline: {
        "prompt",
        "image",
        "original_image",
        "geometry_constraint_mask_image",
        "generative_refinement_mask_image",
        "height",
        "width",
        "padding_mask_crop",
        "strength",
        "num_inference_steps",
        "sigmas",
        "guidance_scale",
        "num_images_per_prompt",
        "generator",
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "max_sequence_length",
        "text_encoder_out_layers",
        "use_dual_mask",
    },
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
        "use_dual_mask",
    },
}


def load_inpainting_pipeline(
    model_name: str = "black-forest-labs/FLUX.1-Fill-dev",
    pipeline_class: type | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    gpu_id: int = 0,
) -> Any:
    """Load and return the inpainting pipeline.

    Call this once and pass the result to ``run_dual_mask_inpainting`` to avoid
    reloading the model on every iteration.

    Parameters
    ----------
    gpu_id
        Physical GPU index passed to ``enable_model_cpu_offload``.  Must match
        the actual device number, not a CUDA_VISIBLE_DEVICES-remapped index.
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
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
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
    original_image_path: str | Path | None = None,
    model_name: str = "black-forest-labs/FLUX.1-Fill-dev",
    prompt_2: str | None = None,
    negative_prompt: str | None = " ",
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
    use_dual_mask: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    pipe: Any | None = None,
    edit_model_prompt_template: str = _DEFAULT_EDIT_MODEL_PROMPT_TEMPLATE,
    edit_models: frozenset[str] | set[str] = _DEFAULT_EDIT_MODELS,
):
    projected_image_path = Path(projected_image_path)
    geometry_constraint_mask_path = Path(geometry_constraint_mask_path)
    generative_refinement_mask_path = Path(generative_refinement_mask_path)
    output_path = Path(output_path)

    original_image = (
        load_image(str(original_image_path)).convert("RGB")
        if original_image_path is not None
        else None
    )
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

    if pipe is None:
        pipe = load_inpainting_pipeline(
            model_name=model_name,
            pipeline_class=MODEL_REGISTRY.get(model_name),
            torch_dtype=torch_dtype,
        )

    generator = torch.Generator("cuda").manual_seed(seed)
    prompt = _apply_prompt_template(prompt, model_name, edit_model_prompt_template, edit_models)

    pipeline_kwargs = _filter_pipeline_kwargs(
        pipe,
        model_name,
        {
            "prompt": prompt,
            "prompt_2": prompt_2,
            "negative_prompt": negative_prompt,
            "image": image,
            "original_image": original_image,
            "geometry_constraint_mask_image": geometry_constraint_mask,
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
            "use_dual_mask": use_dual_mask,
        },
    )
    result = pipe(
        **pipeline_kwargs,
    )

    final_image = result.images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(output_path)

    return final_image
