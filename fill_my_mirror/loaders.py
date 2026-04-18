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
    gt_image_path: str | None = None


@dataclass
class RealImageSample(Sample):
    def __post_init__(self):
        if self.gt_image_path is None:
            self.gt_image_path = self.image_path


@dataclass
class BlenderSample(Sample):
    points: np.ndarray = None
    depth: np.ndarray = None
    intrinsics: np.ndarray = None


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
        sample["image"].save(image_path)
        
        mask_path = tmp_dir / "mask.png"
        sample["mask"].save(mask_path)

        gt_image_path = tmp_dir / "gt_image.png"
        sample["gt_image"].save(gt_image_path)

        prompt = sample.get("prompt") or None
        return RealImageSample(
            image_path=str(image_path),
            mask_path=str(mask_path),
            gt_image_path=str(gt_image_path),
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
        Image.fromarray(np.array(sample["image"])).save(image_path)
        
        mask_path = tmp_dir / "mask.png"
        mirror_mask_arr = (np.array(sample["mask"]) > 127)
        Image.fromarray((mirror_mask_arr.astype(np.uint8) * 255)).save(mask_path)
        
        gt_image_path = tmp_dir / "gt_image.png"
        Image.fromarray(np.array(sample["gt_image"])).save(gt_image_path)

        return BlenderSample(
            image_path=str(image_path),
            mask_path=str(mask_path),
            gt_image_path=str(gt_image_path),
            prompt=sample.get("prompt"),
            points=np.array(sample["points"], dtype=np.float32),
            depth=np.array(sample["depth"], dtype=np.float32),
            intrinsics=np.array(sample["intrinsics"], dtype=np.float32),
        )
