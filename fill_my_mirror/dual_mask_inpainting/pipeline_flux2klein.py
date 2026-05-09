from __future__ import annotations

from typing import Any, Callable

import torch
from diffusers import Flux2KleinInpaintPipeline
from diffusers.utils import logging
from PIL import Image

logger = logging.get_logger(__name__)

_KLEIN_CALL_PARAMS = {
    "prompt",
    "image",
    "image_reference",
    "mask_image",
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
    "output_type",
    "return_dict",
    "attention_kwargs",
    "callback_on_step_end",
    "callback_on_step_end_tensor_inputs",
    "max_sequence_length",
    "text_encoder_out_layers",
}


class DualMaskFlux2KleinInpaintPipeline(Flux2KleinInpaintPipeline):
    """Wraps Flux2KleinInpaintPipeline to accept the dual-mask interface.

    Mapping:
        - ``original_image``                 → ``image``   (scene being inpainted)
        - ``image`` (projected)              → ``image_reference``
        - ``generative_refinement_mask_image`` → ``mask_image``
        - ``geometry_constraint_mask_image`` is accepted but ignored (Klein has no
          geometry-constraint concept; the generative mask covers the full mirror region)
    """

    # FLUX.2-Klein upstream defaults for loggable parameters
    _KLEIN_DEFAULTS = {
        "num_inference_steps": 50,
        "guidance_scale": 8.0,
        "height": None,
        "width": None,
        "padding_mask_crop": None,
        "sigmas": None,
        "num_images_per_prompt": 1,
        "generator": None,
        "latents": None,
        "prompt_embeds": None,
        "negative_prompt_embeds": None,
        "max_sequence_length": 512,
        "text_encoder_out_layers": (9, 18, 27),
    }

    def __call__(
        self,
        prompt: str | list[str] | None = None,
        image: Any | None = None,
        original_image: Any | None = None,
        geometry_constraint_mask_image: Any | None = None,
        generative_refinement_mask_image: Any | None = None,
        height: int | None = None,
        width: int | None = None,
        padding_mask_crop: int | None = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 8.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
        # dual-mask compat shims
        use_dual_mask: bool = True,
        n: float | None = None,
        t_prime: float | None = None,
        **kwargs,
    ):
        if geometry_constraint_mask_image is not None:
            logger.debug(
                "DualMaskFlux2KleinInpaintPipeline: geometry_constraint_mask_image is ignored "
                "(Klein uses a single mask_image)."
            )

        return super().__call__(
            prompt=prompt,
            image=original_image,
            mask_image=generative_refinement_mask_image,
            image_reference=image if use_dual_mask else None,
            strength=strength,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            # Guidance scale is problematic so it's not in use for now.
            # guidance_scale=guidance_scale,
            height=height,
            width=width,
            padding_mask_crop=padding_mask_crop,
            sigmas=sigmas,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
