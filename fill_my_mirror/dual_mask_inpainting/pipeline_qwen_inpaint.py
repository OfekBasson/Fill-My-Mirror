from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from diffusers import QwenImageEditInpaintPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_inpaint import (
    XLA_AVAILABLE,
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.utils import logging

if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm

logger = logging.get_logger(__name__)


class DualMaskInterpolatedQwenInpaintPipeline(QwenImageEditInpaintPipeline):
    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput | None = None,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        geometry_constraint_mask_image: PipelineImageInput = None,
        generative_refinement_mask_image: PipelineImageInput = None,
        masked_image_latents: PipelineImageInput = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        padding_mask_crop: int | None = None,
        strength: float = 0.6,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        n: float = 6.0,
        t_prime: float = 750.0,
    ):
        image_size = image[0].size if isinstance(image, list) else image.size
        calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])

        # height and width are the same as the calculated height and width
        height = calculated_height
        width = calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            image,
            geometry_constraint_mask_image,
            strength,
            height,
            width,
            output_type=output_type,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            padding_mask_crop=padding_mask_crop,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # 3. Preprocess image
        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(geometry_constraint_mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            image = self.image_processor.resize(image, calculated_height, calculated_width)
            original_image = image
            prompt_image = image
            image = self.image_processor.preprocess(
                image,
                height=calculated_height,
                width=calculated_width,
                crops_coords=crops_coords,
                resize_mode=resize_mode,
            )
            image = image.to(dtype=torch.float32)

        has_neg_prompt = negative_prompt is not None or negative_prompt_embeds is not None

        if true_cfg_scale > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt:
            logger.warning(
                " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
            )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=prompt_image,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=prompt_image,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, noise, image_latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        mask_condition = self.mask_processor.preprocess(
            geometry_constraint_mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )

        if masked_image_latents is None:
            masked_image = image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents

        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size,
            num_channels_latents,
            num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        ref_mask_condition = self.mask_processor.preprocess(
            generative_refinement_mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
        ref_masked_image = image * (ref_mask_condition < 0.5)
        ref_mask, ref_masked_image_latents = self.prepare_mask_latents(
            ref_mask_condition,
            ref_masked_image,
            batch_size,
            num_channels_latents,
            num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
            ]
        ] * batch_size

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
            logger.warning(
                f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled."
            )
            guidance = None
        elif not self.transformer.config.guidance_embeds and guidance_scale is None:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        ref_latents = latents.clone()
        T = timesteps[0].float()

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # --- geometry constraint branch ---
                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                latents_dtype = latents.dtype
                step_index_before = self.scheduler._step_index
                latents_geom = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                init_latents_proper = image_latents
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.scale_noise(
                        init_latents_proper, torch.tensor([noise_timestep]), noise
                    )

                latents_geom = (1 - mask) * init_latents_proper + mask * latents_geom

                # --- generative refinement branch (only when t <= t_prime) ---
                if t <= t_prime:
                    ref_latent_model_input = ref_latents
                    if image_latents is not None:
                        ref_latent_model_input = torch.cat([ref_latents, image_latents], dim=1)

                    with self.transformer.cache_context("cond"):
                        ref_noise_pred = self.transformer(
                            hidden_states=ref_latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            encoder_hidden_states=prompt_embeds,
                            img_shapes=img_shapes,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                        ref_noise_pred = ref_noise_pred[:, : ref_latents.size(1)]

                    if do_true_cfg:
                        with self.transformer.cache_context("uncond"):
                            ref_neg_noise_pred = self.transformer(
                                hidden_states=ref_latent_model_input,
                                timestep=timestep / 1000,
                                guidance=guidance,
                                encoder_hidden_states_mask=negative_prompt_embeds_mask,
                                encoder_hidden_states=negative_prompt_embeds,
                                img_shapes=img_shapes,
                                attention_kwargs=self.attention_kwargs,
                                return_dict=False,
                            )[0]
                        ref_neg_noise_pred = ref_neg_noise_pred[:, : ref_latents.size(1)]
                        ref_comb_pred = ref_neg_noise_pred + true_cfg_scale * (ref_noise_pred - ref_neg_noise_pred)
                        ref_cond_norm = torch.norm(ref_noise_pred, dim=-1, keepdim=True)
                        ref_noise_norm = torch.norm(ref_comb_pred, dim=-1, keepdim=True)
                        ref_noise_pred = ref_comb_pred * (ref_cond_norm / ref_noise_norm)

                    self.scheduler._step_index = step_index_before
                    latents_ref = self.scheduler.step(ref_noise_pred, t, ref_latents, return_dict=False)[0]
                    latents_ref = (1 - ref_mask) * init_latents_proper + ref_mask * latents_ref

                    # interpolate between the two latents (norm-preserving)
                    alpha = (t.float() / T).clamp(0, 1)
                    interpolation_scale = alpha.pow(n)

                    mixed = interpolation_scale * latents_geom + (1 - interpolation_scale) * latents_ref

                    reduce_dims = (1, 2)
                    target_norm = torch.linalg.vector_norm(latents_geom, dim=reduce_dims, keepdim=True)
                    mixed_norm = torch.linalg.vector_norm(mixed, dim=reduce_dims, keepdim=True)
                    latents = mixed * (target_norm / (mixed_norm + 1e-8))
                else:
                    latents = latents_geom

                ref_latents = latents.clone()

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)
                    ref_latents = latents.clone()

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

            if padding_mask_crop is not None:
                image = [
                    self.image_processor.apply_overlay(geometry_constraint_mask_image, original_image, i, crops_coords) for i in image
                ]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)