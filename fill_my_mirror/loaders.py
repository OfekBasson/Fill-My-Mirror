import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from datasets import load_dataset


import numpy as np
from PIL import Image


@dataclass
class Sample:
    image_path: str
    mask_path: str
    prompt: str | None


@dataclass
class RealImageSample(Sample):
    pass


@dataclass
class BlenderSample(Sample):
    mirror_mask: np.ndarray  # (800, 800) bool
    points: np.ndarray       # (800, 800, 3) float32
    depth: np.ndarray        # (800, 800) float32
    valid_mask: np.ndarray   # (800, 800) bool
    intrinsics: np.ndarray   # (3, 3) float32


class SampleLoader(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def load(self, index: int) -> Sample: ...


class RealImageSampleLoader(SampleLoader):

    def __init__(self, repo: str):
        self._ds = load_dataset(repo, split="test")

    def __len__(self) -> int:
        return len(self._ds)

    def load(self, index: int) -> RealImageSample:
        sample = self._ds[index]
        tmp_dir = Path(tempfile.mkdtemp(prefix="fill_my_mirror_real_"))

        image_path = tmp_dir / "image.png"
        mask_path = tmp_dir / "mask.png"
        sample["image"].save(image_path)
        sample["mask"].save(mask_path)

        prompt = sample.get("prompt") or None
        return RealImageSample(
            image_path=str(image_path),
            mask_path=str(mask_path),
            prompt=prompt,
        )


class BlenderSampleLoader(SampleLoader):

    def __init__(self, repo: str):
        self._ds = load_dataset(repo, split="train")

    def __len__(self) -> int:
        return len(self._ds)

    def load(self, index: int) -> BlenderSample:
        sample = self._ds[index]
        tmp_dir = Path(tempfile.mkdtemp(prefix="fill_my_mirror_blender_"))

        image_path = tmp_dir / "image.png"
        mask_path = tmp_dir / "mask.png"
        Image.fromarray(np.array(sample["image"])).save(image_path)
        mirror_mask_arr = (np.array(sample["mirror_mask"]) > 127)
        Image.fromarray((mirror_mask_arr.astype(np.uint8) * 255)).save(mask_path)

        return BlenderSample(
            image_path=str(image_path),
            mask_path=str(mask_path),
            prompt=sample.get("prompt") or None,
            mirror_mask=mirror_mask_arr,
            points=np.array(sample["points"], dtype=np.float32),
            depth=np.array(sample["depth"], dtype=np.float32),
            valid_mask=np.array(sample["mask"], dtype=bool),
            intrinsics=np.array(sample["intrinsics"], dtype=np.float32),
        )
