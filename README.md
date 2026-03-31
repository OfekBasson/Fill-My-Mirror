# Fill My Mirror

<p align="center">
  <a href="https://google.com">📄 Paper</a> &nbsp;|&nbsp;
  <a href="TODO">🌐 Project Page</a> &nbsp;|&nbsp;
  <a href="TODO">📦 Dataset</a>
</p>

Official implementation of **Fill My Mirror**.

This repository contains the code for generating consistent reflections in mirrors by combining **geometry estimation**, **projection using Blender**, and **dual-mask diffusion-based inpainting**.

---

# 🛠️ Installation

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

# 🚀 Quick Start

Run the example script:

```bash
bash scripts/run_example.sh
```

Or run directly with your own image and mask:

```bash
python -m fill_my_mirror --image PATH_TO_IMAGE --mask PATH_TO_MASK
```

Or use a sample from the HuggingFace dataset by index:

```bash
python -m fill_my_mirror --hf-index 0
```

---

# ⚙️ Command Line Arguments

## Input (required — choose one)

**Option A — local files:**

```bash
--image PATH_TO_IMAGE
```

Path to the input image.

```bash
--mask PATH_TO_MASK
```

Path to the binary mask of the mirror region.

Both `--image` and `--mask` must be provided together.

---

**Option B — HuggingFace dataset sample:**

```bash
--hf-index INDEX
```

Index of a sample from the HuggingFace dataset (0 to dataset size − 1).  
The dataset repo is read from `hf_dataset_repo` in the config file.  
If the sample includes a caption, it is used as the prompt unless `--prompt` is also given.

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
If provided, it overrides the config file value and any caption from the HuggingFace sample.

```bash
--prompt-2 "SECOND PROMPT"
```

Optional second text prompt.

```bash
--output_path outputs/result.png
```

Path to save the final output image.  
If provided, it overrides the value in the config file.

```bash
--height 1024
--width 1024
```

Desired output resolution. If the requested resolution does not match the input image's aspect ratio, the width is automatically adjusted to fit and a warning is printed.

```bash
--strength 1.0
```

Inpainting strength.

```bash
--num-inference-steps 30
```

Number of diffusion inference steps.

```bash
--guidance-scale 30.0
```

Guidance scale for the diffusion model.

```bash
--num-images-per-prompt 1
```

Number of images to generate per prompt.

```bash
--max-sequence-length 512
```

Maximum text sequence length.

```bash
--seed 0
```

Random seed for reproducibility.

```bash
--n 6.0
```

Power `n` for the alpha^n interpolation.

```bash
--t-prime 750.0
```

First timestep threshold for mask interpolation.

---

# 📁 Default Configuration

The default configuration is stored in `configs/config.yaml`:

```yaml
prompt: "Complete the mirror reflection realistically and consistently with the scene geometry."
default_output_path: "outputs/result.png"
blender_path: "external/blender/blender-4.4.3-linux-x64/blender"
geometry_model_name: "Ruicheng/moge-2-vitl-normal"
inpainting_model_name: "black-forest-labs/FLUX.1-Fill-dev"
hf_dataset_repo: "OfekBassonResearch/Fill-My-Mirror"
```

---

# 📌 Notes

Blender is expected at the path specified in the configuration file.

If Blender is missing, install it with:

```bash
bash scripts/install_blender.sh
```

---

# 💡 Example Commands

Basic example with local files:

```bash
python -m fill_my_mirror \
  --image data/real_images/images/0.png \
  --mask data/real_images/masks/0.png \
  --prompt "A standing mirror reflects a bed with a dotted cover in a cozy bedroom."
```

Example using a HuggingFace dataset sample:

```bash
python -m fill_my_mirror --hf-index 5
```

Example with a custom output path:

```bash
python -m fill_my_mirror \
  --image examples/input/example.jpg \
  --mask examples/input/example_mask.png \
  --output_path outputs/example_result.png
```

Example with a custom prompt:

```bash
python -m fill_my_mirror \
  --image examples/input/example.jpg \
  --mask examples/input/example_mask.png \
  --prompt "Generate a realistic mirror reflection consistent with the scene geometry and lighting."
```

---

# 📦 Dataset

The dataset used in this paper is available on Hugging Face:

**[OfekBasson/fill-my-mirror](https://huggingface.co/datasets/OfekBasson/fill-my-mirror)**

It contains 50 real-world mirror scenes with the following columns per sample:

| Column | Description |
|---|---|
| `image` | Input image (mirror region visible) |
| `mask` | Binary mask of the mirror region |
| `gt_image` | Ground-truth image (mirror filled) |
| `caption` | Text description of the scene |

```python
from datasets import load_dataset

ds = load_dataset("OfekBasson/fill-my-mirror")["test"]
sample = ds[0]
sample["image"].show()
```

To upload the dataset yourself, run:

```bash
pip install datasets huggingface_hub
huggingface-cli login
python scripts/upload_dataset_to_hf.py --repo OfekBasson/fill-my-mirror
```

---

# 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{fill_my_mirror,
  title={Fill My Mirror},
  author={...},
  year={2025}
}
```
