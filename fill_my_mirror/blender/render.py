from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


TEMP_OUTPUT_DIR = Path("temp_outputs")
TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLENDER_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "blender" / "blender_entrypoint.py"

def render_with_blender(
    blender_path: str | Path,
    glb_path: str | Path,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    output_path: str | Path,
    bw_output_path: str | Path,
    depth_output_path: str | Path | None = None,
    tmp_dir: str | Path | None = None,
) -> None:
    blender_path = Path(blender_path)
    glb_path = Path(glb_path)
    output_path = Path(output_path)
    bw_output_path = Path(bw_output_path)
    _tmp_dir = Path(tmp_dir) if tmp_dir is not None else TEMP_OUTPUT_DIR
    _tmp_dir.mkdir(parents=True, exist_ok=True)

    if not blender_path.exists():
        raise FileNotFoundError(f"Blender not found at: {blender_path}")
    if not glb_path.exists():
        raise FileNotFoundError(f"GLB not found at: {glb_path}")
    if not BLENDER_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Blender script not found at: {BLENDER_SCRIPT_PATH}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bw_output_path.parent.mkdir(parents=True, exist_ok=True)
    if depth_output_path is not None:
        depth_output_path = Path(depth_output_path)
        depth_output_path.parent.mkdir(parents=True, exist_ok=True)

    npz_path = _tmp_dir / "blender_render_inputs.npz"
    np.savez(
        npz_path,
        intrinsics=intrinsics,
        image_shape=np.array(image_shape, dtype=np.int32),
    )

    command = [
        str(blender_path),
        "--background",
        "--python",
        str(BLENDER_SCRIPT_PATH),
        "--",
        str(glb_path),
        str(output_path),
        str(bw_output_path),
        str(npz_path),
    ]

    if depth_output_path is not None:
        command.append(str(depth_output_path))

    subprocess.run(command, check=True)