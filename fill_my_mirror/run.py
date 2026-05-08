import argparse
import gc
from pathlib import Path
import torch
import yaml
from fill_my_mirror.geometry import estimate_geometry
from fill_my_mirror.projection import run_projection_single_mirror, run_projection_multiple_mirrors
from fill_my_mirror.geometry import GeometryOutputMultipleMirrors
from fill_my_mirror.dual_mask_inpainting import run_dual_mask_inpainting
from fill_my_mirror.loaders import (
    SampleLoader, RealImageSampleLoader, BlenderSampleLoader,
    MirrorBenchV2SampleLoader, EstimatedGeometrySample,
)
from fill_my_mirror.utils import check_and_fix_aspect_ratio

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def load_config(config_path: Path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Fill My Mirror pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the config YAML file."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the input image. Required unless --hf-index is provided."
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to the mirror mask. Required unless --hf-index or --masks is provided."
    )
    parser.add_argument(
        "--masks",
        type=str,
        nargs="+",
        default=None,
        help="Paths to mirror masks (one per mirror). Mutually exclusive with --mask."
    )
    parser.add_argument(
        "--hf-index",
        type=int,
        default=None,
        help="Index of a sample from the HuggingFace dataset (alternative to --image/--mask)."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for the diffusion model. Overrides the config file."
    )
    parser.add_argument(
        "--prompt-2",
        type=str,
        default=None,
        help="Optional second prompt."
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Inpainting strength."
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=30.0,
        help="Guidance scale."
    )
    parser.add_argument(
        "--num-images-per-prompt",
        type=int,
        default=1,
        help="Number of images per prompt."
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="Maximum text sequence length."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed."
    )
    parser.add_argument(
        "--n",
        type=float,
        default=6.0,
        help="Power n for alpha^n interpolation."
    )
    parser.add_argument(
        "--height",
        type=float,
        default=800,
        help="Height of the desired image."
    )
    parser.add_argument(
        "--width",
        type=float,
        default=800,
        help="Width of the desired image."
    )
    parser.add_argument(
        "--t_prime",
        type=float,
        default=750.0,
        help="First timestep threshold for interpolation."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the final output image. Overrides the config file."
    )
    parser.add_argument(
        "--blender_path",
        type=str,
        default=None,
        help="Path to Blender."
    )
    parser.add_argument(
        "--use-blender-data",
        action="store_true",
        default=False,
        help="Load from the Blender HuggingFace dataset. Requires --hf-index."
    )
    parser.add_argument(
        "--use-mirrorbench-data",
        action="store_true",
        default=False,
        help=(
            "Load from the local MirrorBench V2 (SynMirrorV2) dataset. "
            "Requires --hf-index. The split CSV is fetched from HuggingFace automatically; "
            "tar archives must be extracted into data/mirrorbench_v2/."
        ),
    )
    parser.add_argument(
        "--use-estimated-geometry",
        action="store_true",
        default=False,
        help=(
            "Estimate geometry from the image using the geometry model instead of using "
            "ground-truth geometry. Only relevant for --use-blender-data and --use-mirrorbench-data."
        ),
    )

    args = parser.parse_args()

    using_blender_hf = args.use_blender_data
    using_mirrorbench = args.use_mirrorbench_data
    using_real_hf = args.hf_index is not None and not using_blender_hf and not using_mirrorbench
    using_files = args.image is not None or args.mask is not None or args.masks is not None

    if using_blender_hf and using_mirrorbench:
        parser.error("--use-blender-data and --use-mirrorbench-data are mutually exclusive.")
    if using_blender_hf and args.hf_index is None:
        parser.error("--use-blender-data requires --hf-index.")
    if using_mirrorbench and args.hf_index is None:
        parser.error("--use-mirrorbench-data requires --hf-index.")
    if (using_real_hf or using_blender_hf or using_mirrorbench) and using_files:
        parser.error("Provide either --hf-index OR (--image and --mask/--masks), not both.")
    if not using_real_hf and not using_blender_hf and not using_mirrorbench and not using_files:
        parser.error("Provide either --hf-index or --image with --mask or --masks.")
    if args.mask is not None and args.masks is not None:
        parser.error("--mask and --masks are mutually exclusive.")
    if using_files and args.image is None:
        parser.error("--image is required when using --mask or --masks.")
    if using_files and args.mask is None and args.masks is None:
        parser.error("Either --mask or --masks must be provided with --image.")

    config_path = Path(args.config)
    config = load_config(config_path)

    prompt = args.prompt if args.prompt is not None else config["prompt"]
    output_path = Path(args.output_path) if args.output_path is not None else Path(config["default_output_path"])
    blender_path = Path(args.blender_path) if args.blender_path is not None else Path(config["blender_path"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    loader: SampleLoader | None = None
    if using_mirrorbench:
        loader = MirrorBenchV2SampleLoader()
    elif using_blender_hf:
        loader = BlenderSampleLoader()
    elif using_real_hf:
        loader = RealImageSampleLoader()

    if using_real_hf or using_files:
        if args.use_estimated_geometry:
            print("Note: --use-estimated-geometry has no effect for real images (always estimated).")

    if loader is not None:
        if not (0 <= args.hf_index < len(loader)):
            parser.error(f"--hf-index must be between 0 and {len(loader) - 1}, got {args.hf_index}.")
        use_estimated = True if (using_real_hf or using_files) else args.use_estimated_geometry
        sample = loader.load(args.hf_index, use_estimated_geometry=use_estimated)
        if sample.prompt and args.prompt is None:
            prompt = sample.prompt
    elif args.masks is not None:
        sample = EstimatedGeometrySample(
            image_path=args.image, mask_path=None,
            mask_paths=args.masks, prompt=prompt,
        )
    else:
        sample = EstimatedGeometrySample(image_path=args.image, mask_path=args.mask, prompt=prompt)

    image_path = sample.image_path
    mask_path = sample.mask_path

    width = check_and_fix_aspect_ratio(image_path, int(args.height), int(args.width))

    print("Running Fill My Mirror pipeline")
    print(f"Config: {config_path}")
    print(f"Model: {config['inpainting_model_name']}")
    print(f"Image: {image_path}")
    print(f"Mask: {mask_path}")
    print(f"Prompt: {prompt}")
    print(f"Output: {output_path}")
    print(f"Blender: {blender_path}")

    if not blender_path.exists():
        raise FileNotFoundError(
            f"Blender was not found at: {blender_path}\n"
            "Please install it first with: bash scripts/install_blender.sh"
        )

    geometry = estimate_geometry(sample, config["geometry_model_name"])
    gc.collect()
    torch.cuda.empty_cache()

    if isinstance(geometry, GeometryOutputMultipleMirrors):
        N = len(args.masks)
        constraint_paths = [Path("temp_outputs") / f"geometry_constraint_mask_{i}.png" for i in range(N)]
        projection = run_projection_multiple_mirrors(
            geometry_output=geometry,
            image_path=image_path,
            mirror_mask_paths=[Path(p) for p in args.masks],
            blender_path=blender_path,
            projected_image_path=output_path,
            geometry_constraint_masks_paths=constraint_paths,
        )
        print("Multi-mirror projection complete.")
        print("Projected image:", projection.projected_image_path)
        for i, cp in enumerate(projection.geometry_constraint_masks_paths):
            print(f"  Constraint mask mirror {i}:", cp)
    else:
        projection = run_projection_single_mirror(
            geometry_output=geometry,
            image_path=image_path,
            mirror_mask_path=mask_path,
            blender_path=blender_path,
        )

        run_dual_mask_inpainting(
            prompt=prompt,
            projected_image_path=projection.projected_image_path,
            geometry_constraint_mask_path=projection.geometry_constraint_mask_path,
            generative_refinement_mask_path=mask_path,
            output_path=output_path,
            original_image_path=image_path,
            model_name=config["inpainting_model_name"],
            prompt_2=args.prompt_2,
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt,
            max_sequence_length=args.max_sequence_length,
            seed=args.seed,
            height=args.height,
            width=width,
            n=args.n,
            t_prime=args.t_prime,
        )

        print("Final result saved to:", output_path)


if __name__ == "__main__":
    main()
