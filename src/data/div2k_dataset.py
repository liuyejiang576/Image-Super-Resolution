"""DIV2K datasets for x4 super-resolution experiments."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


def _list_png_files(image_dir: Path) -> List[Path]:
    paths = sorted(image_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG files found under: {image_dir}")
    return paths


def _pil_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 3:
        raise ValueError(f"Expected RGB image with 3 dims, got shape {array.shape}")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _mod_crop(image: Image.Image, scale: int) -> Image.Image:
    width, height = image.size
    width = width - (width % scale)
    height = height - (height % scale)
    return image.crop((0, 0, width, height))


class DIV2KPatchDataset(Dataset):
    """Random HR patch with on-the-fly bicubic LR generation."""

    def __init__(
        self,
        hr_dir: str | Path,
        scale: int = 4,
        hr_patch_size: int = 256,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        if hr_patch_size % scale != 0:
            raise ValueError("hr_patch_size must be divisible by scale.")
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.hr_patch_size = hr_patch_size
        self.augment = augment
        self.seed = seed
        self.hr_paths = _list_png_files(self.hr_dir)

    def __len__(self) -> int:
        return len(self.hr_paths)

    def _random_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width < self.hr_patch_size or height < self.hr_patch_size:
            raise ValueError(
                f"Image {image.size} is smaller than patch {self.hr_patch_size}"
            )
        left = random.randint(0, width - self.hr_patch_size)
        top = random.randint(0, height - self.hr_patch_size)
        return image.crop((left, top, left + self.hr_patch_size, top + self.hr_patch_size))

    def _augment_pair(self, hr: Image.Image, lr: Image.Image) -> tuple[Image.Image, Image.Image]:
        if not self.augment:
            return hr, lr
        if random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
            lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
        # Use exact 90-degree transposes to avoid interpolation artifacts.
        rotations = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        op = rotations[random.randint(0, 3)]
        if op is not None:
            hr = hr.transpose(op)
            lr = lr.transpose(op)
        return hr, lr

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        hr_path = self.hr_paths[index]
        with Image.open(hr_path) as image:
            hr_img = image.convert("RGB")

        hr_patch = self._random_crop(hr_img)
        lr_size = self.hr_patch_size // self.scale
        lr_patch = hr_patch.resize((lr_size, lr_size), Image.BICUBIC)
        hr_patch, lr_patch = self._augment_pair(hr_patch, lr_patch)

        return {
            "hr": _pil_to_tensor(hr_patch),
            "lr": _pil_to_tensor(lr_patch),
            "path": str(hr_path),
        }


class DIV2KFullImageDataset(Dataset):
    """Validation dataset with full-image HR/LR pairs."""

    def __init__(self, hr_dir: str | Path, scale: int = 4) -> None:
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.hr_paths = _list_png_files(self.hr_dir)

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        hr_path = self.hr_paths[index]
        with Image.open(hr_path) as image:
            hr_img = _mod_crop(image.convert("RGB"), self.scale)

        width, height = hr_img.size
        lr_img = hr_img.resize((width // self.scale, height // self.scale), Image.BICUBIC)
        return {
            "hr": _pil_to_tensor(hr_img),
            "lr": _pil_to_tensor(lr_img),
            "path": str(hr_path),
        }
