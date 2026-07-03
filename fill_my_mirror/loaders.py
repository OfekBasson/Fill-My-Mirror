import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import h5py
import json
import numpy as np
import pandas as pd
from PIL import Image

REAL_IMAGES_HF_REPO = "OfekBassonResearch/Fill-My-Mirror"
BLENDER_HF_REPO = "OfekBassonResearch/Fill-My-Mirror-Blender"
MIRRORBENCH_V2_HF_REPO = "ankitIIsc/SynMirrorV2"
MIRRORBENCH_V2_DATA_ROOT = Path("data/mirrorbench_v2")


@dataclass
class Sample:
    image_path: str
    mask_path: str | None
    prompt: str | None
    gt_image_path: str | None = None
    mask_paths: list[str] = field(default_factory=list)


@dataclass
class EstimatedGeometrySample(Sample):
    def __post_init__(self):
        if self.image_path is not None and self.gt_image_path is None:
            self.gt_image_path = self.image_path


@dataclass
class GTGeometrySample(Sample):
    points: np.ndarray = None
    depth: np.ndarray = None
    intrinsics: np.ndarray = None


@dataclass
class DepthDegradedSample(GTGeometrySample):
    """GTGeometrySample annotated with depth-degradation parameters.

    lam=0 → pure GT depth; lam=1 → MoGe estimate; lam>1 → extrapolated.
    image_id is the stable dataset index used to cache the MoGe result across
    multiple lambda values for the same image.
    """
    lam: float = 0.0
    image_id: int = 0


class SampleLoader(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def load(self, index: int, use_estimated_geometry: bool = False) -> Sample: ...


class RealImageSampleLoader(SampleLoader):

    def __init__(self):
        self._ds = load_dataset(REAL_IMAGES_HF_REPO, split="test")

    def __len__(self) -> int:
        return len(self._ds)

    def load(self, index: int, use_estimated_geometry: bool = True) -> EstimatedGeometrySample:
        sample = self._ds[index]
        tmp_dir = Path(tempfile.mkdtemp(prefix="fill_my_mirror_real_"))

        image_path = tmp_dir / "image.png"
        sample["image"].save(image_path)
        
        mask_path = tmp_dir / "mask.png"
        sample["mask"].save(mask_path)

        gt_image_path = tmp_dir / "gt_image.png"
        sample["gt_image"].save(gt_image_path)

        prompt = sample.get("prompt") or None
        return EstimatedGeometrySample(
            image_path=str(image_path),
            mask_path=str(mask_path),
            gt_image_path=str(gt_image_path),
            prompt=prompt,
        )


class BlenderSampleLoader(SampleLoader):

    def __init__(self):
        self._ds = load_dataset(BLENDER_HF_REPO, split="train")

    def __len__(self) -> int:
        return len(self._ds)

    def load(self, index: int, use_estimated_geometry: bool = False) -> Sample:
        sample = self._ds[index]
        tmp_dir = Path(tempfile.mkdtemp(prefix="fill_my_mirror_blender_"))

        image_path = tmp_dir / "image.png"
        Image.fromarray(np.array(sample["image"])).save(image_path)

        mask_path = tmp_dir / "mask.png"
        mirror_mask_arr = (np.array(sample["mask"]) > 127)
        Image.fromarray((mirror_mask_arr.astype(np.uint8) * 255)).save(mask_path)

        gt_image_path = tmp_dir / "gt_image.png"
        Image.fromarray(np.array(sample["gt_image"])).save(gt_image_path)

        if use_estimated_geometry:
            return EstimatedGeometrySample(
                image_path=str(image_path),
                mask_path=str(mask_path),
                gt_image_path=str(gt_image_path),
                prompt=sample.get("prompt"),
            )
        return GTGeometrySample(
            image_path=str(image_path),
            mask_path=str(mask_path),
            gt_image_path=str(gt_image_path),
            prompt=sample.get("prompt"),
            points=np.array(sample["points"], dtype=np.float32),
            depth=np.array(sample["depth"], dtype=np.float32),
            intrinsics=np.array(sample["intrinsics"], dtype=np.float32),
        )


def _decode_cam_states(cam_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (cam2world 4×4, cam_K 3×3) from the HDF5 cam_states byte array."""
    data = json.loads(cam_states.tobytes().decode("utf-8"))
    return np.array(data["cam2world"], dtype=np.float64), np.array(data["cam_K"], dtype=np.float64)


class MirrorBenchV2SampleLoader(SampleLoader):
    """Load samples from a locally extracted SynMirrorV2 (MirrorBench) dataset.

    The split CSV (test.csv) is downloaded automatically from HuggingFace.
    The HDF5 files must be extracted from the dataset tar archives into
    MIRRORBENCH_V2_DATA_ROOT (data/mirrorbench_v2/).
    """

    def __init__(self):
        csv_path = hf_hub_download(
            repo_id=MIRRORBENCH_V2_HF_REPO,
            filename="test.csv",
            repo_type="dataset",
        )
        self._df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self._df)

    def load(self, index: int, use_estimated_geometry: bool = False) -> Sample:
        row = self._df.iloc[index]
        hdf5_path = MIRRORBENCH_V2_DATA_ROOT / row["path"]

        with h5py.File(hdf5_path, "r") as f:
            colors = np.array(f["colors"], dtype=np.uint8)
            segmaps = np.array(f["category_id_segmaps"], dtype=np.uint8)
            depth = np.array(f["depth"], dtype=np.float32)
            cam_states = np.array(f["cam_states"])

        H, W = depth.shape

        mirror_mask = (segmaps == 1)
        mask_arr = (mirror_mask.astype(np.uint8) * 255)

        # gt_image is the full rendered scene; image has the mirror region zeroed out.
        gt_image_arr = colors
        image_arr = colors.copy()
        image_arr[mirror_mask] = 0

        tmp_dir = Path(tempfile.mkdtemp(prefix="fill_my_mirror_mirrorbench_"))

        image_path = tmp_dir / "image.png"
        Image.fromarray(image_arr).save(image_path)

        mask_path = tmp_dir / "mask.png"
        Image.fromarray(mask_arr).save(mask_path)

        gt_image_path = tmp_dir / "gt_image.png"
        Image.fromarray(gt_image_arr).save(gt_image_path)

        caption = row.get("auto_caption")
        prompt = f"A perfect plane mirror reflection of {caption}." if isinstance(caption, str) else None

        if use_estimated_geometry:
            return EstimatedGeometrySample(
                image_path=str(image_path),
                mask_path=str(mask_path),
                gt_image_path=str(gt_image_path),
                prompt=prompt,
            )

        _, cam_K = _decode_cam_states(cam_states)
        fx, fy = float(cam_K[0, 0]), float(cam_K[1, 1])
        cx, cy = float(cam_K[0, 2]), float(cam_K[1, 2])

        # Normalize intrinsics to the [fx/W, cx/W, cy/H] convention used by MoGe / BlenderGeometryProcessor.
        intrinsics = np.array(
            [[fx / W, 0.0,    cx / W],
             [0.0,    fy / H, cy / H],
             [0.0,    0.0,    1.0   ]],
            dtype=np.float32,
        )

        # Unproject depth to camera-space 3-D points, then apply the Blender sign flip.
        ys, xs = np.mgrid[0:H, 0:W]
        X = (xs - cx) / fx * depth
        Y = (ys - cy) / fy * depth
        Z = depth
        points = np.stack([X, Y, Z], axis=-1).astype(np.float32)
        points *= np.array([-1.0, -1.0, 1.0], dtype=np.float32)

        return GTGeometrySample(
            image_path=str(image_path),
            mask_path=str(mask_path),
            gt_image_path=str(gt_image_path),
            prompt=prompt,
            points=points,
            depth=depth,
            intrinsics=intrinsics,
        )
