# Fill My Mirror

Official implementation of **Fill My Mirror**.

This repository contains the code for generating consistent reflections in mirrors by combining **geometry estimation**, **projection using Blender**, and **diffusion-based inpainting**.

---

# Installation

The following instructions were tested on **Linux with Python 3.10**.

## 1. Clone the repository

Clone the repository **with submodules** to also download the MoGe dependency.

git clone --recurse-submodules https://github.com/YOUR_USERNAME/fill-my-mirror.git
cd fill-my-mirror

If you already cloned without submodules, run:

git submodule update --init --recursive

---

## 2. Create a Conda environment

Create a clean environment to avoid dependency conflicts.

conda create -n fill-my-mirror python=3.10
conda activate fill-my-mirror

---

## 3. Install Python dependencies

pip install -r requirements.txt
pip install -e .

`pip install -e .` installs the project as a local Python package so it can be imported from anywhere.

---

## 4. Install Blender

This project requires **Blender** for scene rendering.

Run the provided installation script:

bash scripts/install_blender.sh

This downloads and extracts Blender to:

external/blender/

You can verify the installation with:

external/blender/blender-4.4.3-linux-x64/blender --version

---

# Example Usage

Run the example script:

bash scripts/run_example.sh

Or run the pipeline directly:

python -m fill_my_mirror.run

---

# Repository Structure

fill-my-mirror/
│
├── fill_my_mirror/        # main source code
├── third_party/MoGe/      # MoGe geometry estimation submodule
├── scripts/               # helper scripts
├── external/blender/      # Blender installation
├── configs/               # configuration files
├── examples/              # example inputs and outputs
└── data/                  # datasets (not tracked by git)

---

# Citation

If you use this code in your research, please cite:

@article{fill_my_mirror,
  title={Fill My Mirror},
  author={...},
  year={2025}
}