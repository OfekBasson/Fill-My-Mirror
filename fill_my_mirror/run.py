import argparse
from pathlib import Path
import yaml
from fill_my_mirror.geometry import estimate_geometry


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
        required=True,
        help="Path to the input image."
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to the mirror mask."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for the diffusion model. Overrides the config file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the final output image. Overrides the config file."
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    prompt = args.prompt if args.prompt is not None else config["prompt"]
    output_path = Path(args.output) if args.output is not None else Path(config["output"])
    blender_path = Path(config["blender_path"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Running Fill My Mirror pipeline")
    print(f"Config: {config_path}")
    print(f"Image: {args.image}")
    print(f"Mask: {args.mask}")
    print(f"Prompt: {prompt}")
    print(f"Output: {output_path}")
    print(f"Blender: {blender_path}")

    if not blender_path.exists():
        raise FileNotFoundError(
            f"Blender was not found at: {blender_path}\n"
            "Please install it first with: bash scripts/install_blender.sh"
        )

    geometry = estimate_geometry(
        image_path=args.image,
        model_name=config["geometry_model_name"],
    )

    print("Mesh saved to:", geometry.mesh_path)
    # projection
    # diffusion
    # save result


if __name__ == "__main__":
    main()