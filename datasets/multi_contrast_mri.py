"""Dataset loader for the preprocessed BraTS2020 and fastMRI pairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict

import cv2
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetLayout:
    target_hr: str
    reference_hr: str
    target_lr: str


DATASET_LAYOUTS: Dict[str, DatasetLayout] = {
    "brats2020": DatasetLayout(
        target_hr="oriT2",
        reference_hr="oriT1",
        target_lr="orLRbicT2",
    ),
    "fastmri": DatasetLayout(
        target_hr="oriFSPD",
        reference_hr="oriPD",
        target_lr="orLRbicFSPD",
    ),
}


def random_rigid_misalignment(
    image: np.ndarray,
    max_rotation_deg: float = 5.0,
    max_translation_px: float = 5.0,
) -> np.ndarray:
    """Apply a random in-plane rigid transform to the reference image only."""
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected HxW or HxWxC image, got {image.shape}")
    height, width = image.shape[:2]
    angle = random.uniform(-max_rotation_deg, max_rotation_deg)
    tx = random.uniform(-max_translation_px, max_translation_px)
    ty = random.uniform(-max_translation_px, max_translation_px)
    matrix = cv2.getRotationMatrix2D(
        ((width - 1) / 2.0, (height - 1) / 2.0),
        angle,
        1.0,
    )
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    return cv2.warpAffine(
        image,
        matrix,
        dsize=(width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _image_to_tensor(image: np.ndarray) -> Tensor:
    if image.ndim == 2:
        image = image[..., None]
    array = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(array).mul_(2.0).sub_(1.0)


def _load_image(path: Path, channels: int) -> np.ndarray:
    mode = "L" if channels == 1 else "RGB"
    return np.asarray(Image.open(path).convert(mode))


class MultiContrastMRIDataset(Dataset):
    """Load target/reference pairs from the preprocessing layout used in the paper.

    Returned dictionary fields:
        target_lr: LR target input, normalized to [-1, 1]
        reference_hr: HR reference input, normalized to [-1, 1]
        target_hr: HR target ground truth, normalized to [-1, 1]
        name: source filename
    """

    VALID_CONDITIONS = {"clean", "mis3", "mis6", "mis9", "native"}

    def __init__(
        self,
        root: str | Path,
        dataset: str,
        scale: int = 4,
        split: str = "train",
        condition: str = "clean",
        channels: int = 3,
        train_misalignment: bool = True,
        misalignment_probability: float = 0.5,
        max_rotation_deg: float = 5.0,
        max_translation_px: float = 5.0,
    ) -> None:
        super().__init__()
        dataset = dataset.lower()
        condition = condition.lower()
        if dataset not in DATASET_LAYOUTS:
            raise ValueError(f"Unsupported dataset {dataset!r}; choose from {sorted(DATASET_LAYOUTS)}")
        if condition not in self.VALID_CONDITIONS:
            raise ValueError(f"Unsupported condition {condition!r}")
        if scale <= 0:
            raise ValueError("scale must be positive")
        if channels not in {1, 3}:
            raise ValueError("channels must be 1 or 3")
        if not 0.0 <= misalignment_probability <= 1.0:
            raise ValueError("misalignment_probability must be in [0,1]")

        self.root = Path(root)
        self.dataset = dataset
        self.scale = int(scale)
        self.split = split
        self.condition = condition
        self.channels = int(channels)
        self.is_train = split == "train"
        self.train_misalignment = bool(train_misalignment and self.is_train)
        self.misalignment_probability = float(misalignment_probability)
        self.max_rotation_deg = float(max_rotation_deg)
        self.max_translation_px = float(max_translation_px)

        layout = DATASET_LAYOUTS[dataset]
        split_root = self._resolve_split_root()
        reference_folder = self._reference_folder(layout.reference_hr)
        self.target_hr_dir = split_root / layout.target_hr
        self.reference_hr_dir = split_root / reference_folder
        self.target_lr_dir = split_root / layout.target_lr / f"x{self.scale}"

        for directory in (
            self.target_hr_dir,
            self.reference_hr_dir,
            self.target_lr_dir,
        ):
            if not directory.is_dir():
                raise FileNotFoundError(f"Required dataset directory not found: {directory}")

        target_names = sorted(path.name for path in self.target_hr_dir.iterdir() if path.is_file())
        if not target_names:
            raise RuntimeError(f"No images found in {self.target_hr_dir}")

        self.samples = []
        for name in target_names:
            target_hr = self.target_hr_dir / name
            reference_hr = self.reference_hr_dir / name
            target_lr = self.target_lr_dir / name
            missing = [path for path in (reference_hr, target_lr) if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Pairing by filename failed for {name!r}; missing: {missing}"
                )
            self.samples.append((target_hr, reference_hr, target_lr, name))

    def _resolve_split_root(self) -> Path:
        if self.is_train:
            return self.root / "train"
        if self.dataset == "fastmri" or self.condition == "native":
            return self.root / self.split
        robustness_root = self.root / "val_idm"
        if robustness_root.is_dir():
            return robustness_root
        if self.condition != "clean":
            raise FileNotFoundError(
                f"Condition {self.condition!r} requires {robustness_root}"
            )
        return self.root / self.split

    def _reference_folder(self, base_folder: str) -> str:
        if self.dataset == "fastmri" or self.condition in {"clean", "native"}:
            return base_folder
        return f"{base_folder}_{self.condition}"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        target_hr_path, reference_hr_path, target_lr_path, name = self.samples[index]
        target_hr = _load_image(target_hr_path, self.channels)
        reference_hr = _load_image(reference_hr_path, self.channels)
        reference_native = reference_hr.copy()
        target_lr = _load_image(target_lr_path, self.channels)

        if self.train_misalignment and random.random() < self.misalignment_probability:
            reference_hr = random_rigid_misalignment(
                reference_hr,
                max_rotation_deg=self.max_rotation_deg,
                max_translation_px=self.max_translation_px,
            )

        return {
            "target_hr": _image_to_tensor(target_hr),
            "reference_hr": _image_to_tensor(reference_hr),
            "reference_native": _image_to_tensor(reference_native),
            "target_lr": _image_to_tensor(target_lr),
            "name": name,
        }
