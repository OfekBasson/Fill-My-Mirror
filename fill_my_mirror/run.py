import argparse
from pathlib import Path
import yaml
from fill_my_mirror.geometry import estimate_geometry
from fill_my_mirror.projection import run_projection
from fill_my_mirror.dual_mask_inpainting import run_dual_mask_inpainting
from fill_my_mirror.loaders import SampleLoader, RealImageSampleLoader, BlenderSampleLoader, RealImageSample
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
        help="Path to the mirror mask. Required unless --hf-index is provided."
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
        default=1024,
        help="Height of the desired image."
    )
    parser.add_argument(
        "--width",
        type=float,
        default=1024,
        help="Width of the desired image."
    )
    parser.add_argument(
        "--t-prime",
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

    args = parser.parse_args()

    using_blender_hf = args.use_blender_data
    using_real_hf = args.hf_index is not None and not using_blender_hf
    using_files = args.image is not None or args.mask is not None

    if using_blender_hf and args.hf_index is None:
        parser.error("--use-blender-data requires --hf-index.")
    if (using_real_hf or using_blender_hf) and using_files:
        parser.error("Provide either --hf-index OR (--image and --mask), not both.")
    if not using_real_hf and not using_blender_hf and not using_files:
        parser.error("Provide either --hf-index or both --image and --mask.")
    if using_files and (args.image is None or args.mask is None):
        parser.error("Both --image and --mask must be provided together.")

    config_path = Path(args.config)
    config = load_config(config_path)

    prompt = args.prompt if args.prompt is not None else config["prompt"]
    output_path = Path(args.output_path) if args.output_path is not None else Path(config["default_output_path"])
    blender_path = Path(args.blender_path) if args.blender_path is not None else Path(config["blender_path"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    loader: SampleLoader | None = None
    if using_blender_hf:
        loader = BlenderSampleLoader(config["hf_blender_dataset_repo"])
    elif using_real_hf:
        loader = RealImageSampleLoader(config["hf_dataset_repo"])

    if loader is not None:
        if not (0 <= args.hf_index < len(loader)):
            parser.error(f"--hf-index must be between 0 and {len(loader) - 1}, got {args.hf_index}.")
        sample = loader.load(args.hf_index)
        if sample.prompt and args.prompt is None:
            prompt = sample.prompt
    else:
        sample = RealImageSample(image_path=args.image, mask_path=args.mask, prompt=None)
    
    image_path = sample.image_path
    mask_path = sample.mask_path

    width = check_and_fix_aspect_ratio(image_path, int(args.height), int(args.width))

    print("Running Fill My Mirror pipeline")
    print(f"Config: {config_path}")
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

    print("Mesh saved to:", geometry.mesh_path)

    projection = run_projection(
        geometry_output=geometry,
        image_path=image_path,
        mirror_mask_path=mask_path,
        blender_path=config["blender_path"],
    )

    run_dual_mask_inpainting(
        prompt=prompt,
        projected_image_path=projection.projected_image_path,
        geometry_constraint_mask_path=projection.geometry_constraint_mask_path,
        generative_refinement_mask_path=mask_path,
        output_path=output_path,
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
