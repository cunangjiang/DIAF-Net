"""Small utilities shared by training and evaluation."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from skimage.metrics import structural_similarity


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def denormalize(x: Tensor) -> Tensor:
    return x.detach().float().add(1.0).mul(0.5).clamp(0.0, 1.0)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _to_hwc(image: Tensor) -> np.ndarray:
    image = denormalize(image).cpu().numpy()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {image.shape}")
    return np.transpose(image, (1, 2, 0))


def image_metrics(prediction: Tensor, target: Tensor) -> Dict[str, float]:
    """Compute PSNR, SSIM and RMSE for one normalized CHW image pair."""
    pred = _to_hwc(prediction)
    gt = _to_hwc(target)
    mse = float(np.mean((pred - gt) ** 2))
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    rmse = float(np.sqrt(mse))
    channel_axis = -1 if pred.shape[-1] > 1 else None
    if channel_axis is None:
        pred_ssim = pred[..., 0]
        gt_ssim = gt[..., 0]
    else:
        pred_ssim = pred
        gt_ssim = gt
    ssim = float(
        structural_similarity(
            gt_ssim,
            pred_ssim,
            data_range=1.0,
            channel_axis=channel_axis,
        )
    )
    return {"psnr": psnr, "ssim": ssim, "rmse": rmse}


def save_image(tensor: Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _to_hwc(tensor)
    image_u8 = np.rint(image * 255.0).clip(0, 255).astype(np.uint8)
    if image_u8.shape[-1] == 1:
        image_u8 = image_u8[..., 0]
    Image.fromarray(image_u8).save(path)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)
