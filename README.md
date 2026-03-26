# Fill My Mirror

Official implementation of **Fill My Mirror**.

<!-- TODOs -->
<!-- Add link to the paper and project page (beautiful like in MoGe repo) -->
<!-- Add demo -->
<!-- Add emojis -->
<!-- Add link to dataset download -->
<!-- Add minimal code example -->
<!-- Remove all hardcoded text (only in the yaml file is fine) -->
<!-- Add seed option -->
<!-- Remove masked_image_latents from the __call__ function (or do 2 latents to pass, one for each mask) -->
This repository contains the code for generating consistent reflections in mirrors by combining **geometry estimation**, **projection using Blender**, and **dual-mask diffusion-based inpainting**.

---

# Installation

The following instructions were tested on **Linux with Python 3.10**.

## 1. Clone the repository

Clone the repository **with submodules** to also download the MoGe dependency.

```bash
git clone --recurse-submodules https://github.com/OfekBasson/Fill-My-Mirror.git
cd Fill-My-Mirror
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

---

## 2. Create a Conda environment

Create a clean environment to avoid dependency conflicts.

```bash
conda create -n fill-my-mirror python=3.10
conda activate fill-my-mirror
```

---

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
pip install -e third_party/MoGe
pip install -e .
```

---

## 4. Install Blender

This project requires **Blender** for scene rendering.

Run the provided installation script:

```bash
bash scripts/install_blender.sh
```

This downloads and extracts Blender to:

```
external/blender/
```

You can verify the installation with:

```bash
external/blender/blender-4.4.3-linux-x64/blender --version
```

---

# Example Usage

Run the example script:

```bash
bash scripts/run_example.sh
```

Or run the pipeline directly:

```bash
python -m fill_my_mirror
```

---

# Command Line Arguments

The main pipeline can be run with:

```bash
python -m fill_my_mirror --image PATH_TO_IMAGE --mask PATH_TO_MASK
```

## Required arguments

```bash
--image PATH_TO_IMAGE
```

Path to the input image.

```bash
--mask PATH_TO_MASK
```

Path to the binary mask of the mirror region.

---

## Optional arguments

```bash
--config configs/config.yaml
```

Path to a YAML configuration file.  
Default: `configs/config.yaml`

```bash
--prompt "TEXT PROMPT"
```

Text prompt for the diffusion model.  
If provided, it overrides the value in the config file.

```bash
--output outputs/result.png
```

Path to save the final output image.  
If provided, it overrides the value in the config file.

---

# Default Configuration

The default configuration is stored in:

```
configs/config.yaml
```

Example configuration:

```bash
prompt: "Complete the mirror reflection realistically and consistently with the scene geometry, preserving the visible scene content, perspective, lighting, and object identity."
output: "outputs/result.png"
blender_path: "external/blender/blender-4.4.3-linux-x64/blender"
geometry_model_name: "Ruicheng/moge-2-vitl-normal"
```

---

# Notes

Blender is expected at the path specified in the configuration file.

If Blender is missing, install it with:

```bash
bash scripts/install_blender.sh
```

---

# Example Commands

Basic example:

```bash
python -m fill_my_mirror \
--image data/real_images/images/0.png \ --mask data/real_images/masks/0.png --prompt "A standing mirror reflects a bed with a dotted cover in a cozy bedroom. Above the bed is a brown rattan headboard and a window with dark gray aluminum."
```

Example with custom output path:

```bash
python -m fill_my_mirror \
  --image examples/input/example.jpg \
  --mask examples/input/example_mask.png \
  --output outputs/example_result.png
```

Example with a custom prompt:

```bash
python -m fill_my_mirror \
  --image examples/input/example.jpg \
  --mask examples/input/example_mask.png \
  --prompt "Generate a realistic mirror reflection consistent with the scene geometry and lighting."
```

---

# Citation

If you use this code in your research, please cite:

```bash
@article{fill_my_mirror,
  title={Fill My Mirror},
  author={...},
  year={2025}
}
```