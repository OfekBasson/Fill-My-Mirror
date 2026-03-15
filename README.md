# Fill My Mirror

Official implementation of **Fill My Mirror**.

This repository contains the code for generating consistent reflections in mirrors by combining **geometry estimation**, **projection using Blender**, and **dual-mask diffusion-based inpainting**.

---

# Installation

The following instructions were tested on **Linux with Python 3.10**.

## 1. Clone the repository

Clone the repository **with submodules** to also download the MoGe dependency.
```bash
git clone --recurse-submodules https://github.com/OfekBasson/Fill-My-Mirror.git
cd fill-my-mirror
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

external/blender/

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
python -m fill_my_mirror.run
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