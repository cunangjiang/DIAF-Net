"""Dispersion-Guided Coarse-to-Fine Alignment (DGCA)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d
except Exception as exc:  # pragma: no cover - installation dependent
    deform_conv2d = None
    _DEFORM_IMPORT_ERROR = exc
else:
    _DEFORM_IMPORT_ERROR = None


def _resize_like(x: Tensor, reference: Tensor) -> Tensor:
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    return F.interpolate(
        x,
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def flow_warp(
    feature: Tensor,
    flow_xy: Tensor,
    padding_mode: str = "border",
) -> Tensor:
    """Backward-warp a BCHW feature using feature-pixel displacements (dx, dy)."""
    if feature.ndim != 4:
        raise ValueError(f"feature must be BCHW, got {tuple(feature.shape)}")
    if flow_xy.ndim != 4 or flow_xy.shape[1] != 2:
        raise ValueError(f"flow_xy must be [B,2,H,W], got {tuple(flow_xy.shape)}")
    if flow_xy.shape[0] != feature.shape[0] or flow_xy.shape[-2:] != feature.shape[-2:]:
        raise ValueError("flow_xy must match the feature batch and spatial dimensions")

    _, _, height, width = feature.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=feature.device, dtype=feature.dtype),
        torch.arange(width, device=feature.device, dtype=feature.dtype),
        indexing="ij",
    )
    base_grid = torch.stack((xx, yy), dim=0).unsqueeze(0)
    sample_grid = base_grid + flow_xy
    grid_x = 2.0 * sample_grid[:, 0] / max(width - 1, 1) - 1.0
    grid_y = 2.0 * sample_grid[:, 1] / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)

    return F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


def effective_displacement(
    base_offset: Tensor,
    detail_offset: Tensor,
    dcn_weight: Tensor,
    kernel_size: int,
) -> Tensor:
    """Combine coarse flow and the DCN-weighted mean detail displacement."""
    if base_offset.ndim != 4 or base_offset.shape[1] != 2:
        raise ValueError("base_offset must be [B,2,H,W]")
    kernel_points = int(kernel_size) ** 2
    if detail_offset.ndim != 4 or detail_offset.shape[1] != 2 * kernel_points:
        raise ValueError(
            f"detail_offset must have {2 * kernel_points} channels for "
            f"kernel_size={kernel_size}"
        )

    batch, _, height, width = detail_offset.shape
    # torchvision stores each deformable-kernel pair as (dy, dx).
    detail_yx = detail_offset.view(batch, kernel_points, 2, height, width)
    detail_y = detail_yx[:, :, 0]
    detail_x = detail_yx[:, :, 1]

    importance = dcn_weight.detach().abs().mean(dim=(0, 1)).reshape(
        1, kernel_points, 1, 1
    )
    importance = importance / importance.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mean_x = (detail_x * importance).sum(dim=1, keepdim=True)
    mean_y = (detail_y * importance).sum(dim=1, keepdim=True)
    return base_offset + torch.cat((mean_x, mean_y), dim=1)


class DispersionOffsetPredictor(nn.Module):
    """Predict a bounded base flow and an SFDM-modulated deformable offset."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        max_base_offset: float = 4.0,
        max_detail_offset: float = 2.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.kernel_size = int(kernel_size)
        self.offset_channels = 2 * self.kernel_size * self.kernel_size
        self.max_base_offset = float(max_base_offset)
        self.max_detail_offset = float(max_detail_offset)

        self.body = nn.Sequential(
            nn.Conv2d(2 * channels + 1, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.base_head = nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1)
        self.detail_head = nn.Conv2d(
            hidden_channels,
            self.offset_channels,
            kernel_size=3,
            padding=1,
        )
        gate_hidden = max(hidden_channels // 4, 8)
        self.gate_head = nn.Sequential(
            nn.Conv2d(1, gate_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(gate_hidden, self.offset_channels, kernel_size=3, padding=1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.base_head.weight)
        nn.init.zeros_(self.base_head.bias)
        nn.init.zeros_(self.detail_head.weight)
        nn.init.zeros_(self.detail_head.bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)

    def forward(
        self,
        target_feature: Tensor,
        reference_feature: Tensor,
        sfdm: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if target_feature.shape != reference_feature.shape:
            raise ValueError("target_feature and reference_feature must have the same shape")
        sfdm = _resize_like(sfdm, target_feature)
        joint = self.body(torch.cat((target_feature, reference_feature, sfdm), dim=1))

        base_offset = self.max_base_offset * torch.tanh(self.base_head(joint))
        detail_offset = self.max_detail_offset * torch.tanh(self.detail_head(joint))
        dispersion_gate = 1.0 + torch.tanh(self.gate_head(sfdm))
        return base_offset, detail_offset * dispersion_gate


class DispersionGuidedCoarseToFineAlignment(nn.Module):
    """Correct dominant displacement, then refine local deformation with DCN."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        kernel_size: int = 3,
        max_base_offset: float = 4.0,
        max_detail_offset: float = 2.0,
    ) -> None:
        super().__init__()
        if deform_conv2d is None:
            raise ImportError(
                "DGCA requires torchvision.ops.deform_conv2d. Install matching "
                "PyTorch and torchvision builds."
            ) from _DEFORM_IMPORT_ERROR
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        hidden_channels = hidden_channels or max(channels // 2, 16)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.offset_predictor = DispersionOffsetPredictor(
            channels=channels,
            hidden_channels=hidden_channels,
            kernel_size=self.kernel_size,
            max_base_offset=max_base_offset,
            max_detail_offset=max_detail_offset,
        )
        self.dcn_weight = nn.Parameter(self._identity_weight(channels, self.kernel_size))
        self.dcn_bias = nn.Parameter(torch.zeros(channels))

    @staticmethod
    def _identity_weight(channels: int, kernel_size: int) -> Tensor:
        weight = torch.zeros(channels, channels, kernel_size, kernel_size)
        center = kernel_size // 2
        index = torch.arange(channels)
        weight[index, index, center, center] = 1.0
        return weight

    def forward(
        self,
        target_feature: Tensor,
        reference_feature: Tensor,
        sfdm: Tensor,
        return_aux: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        if target_feature.shape != reference_feature.shape:
            raise ValueError("target_feature and reference_feature must have the same shape")
        if sfdm.ndim != 4 or sfdm.shape[1] != 1:
            raise ValueError("sfdm must be [B,1,H,W]")

        sfdm = _resize_like(sfdm, target_feature)
        base_offset, detail_offset = self.offset_predictor(
            target_feature,
            reference_feature,
            sfdm,
        )
        coarse_aligned = flow_warp(reference_feature, base_offset)
        try:
            aligned = deform_conv2d(
                input=coarse_aligned,
                offset=detail_offset,
                weight=self.dcn_weight,
                bias=self.dcn_bias,
                stride=(1, 1),
                padding=(self.padding, self.padding),
                dilation=(1, 1),
                mask=None,
            )
        except RuntimeError as exc:
            if "custom C++ ops" in str(exc):
                raise RuntimeError(
                    "torchvision deformable-convolution operators are unavailable. "
                    "Reinstall a torchvision build matching the installed PyTorch."
                ) from exc
            raise

        if not return_aux:
            return aligned
        return aligned, {
            "base_offset": base_offset,
            "detail_offset": detail_offset,
            "effective_offset": effective_displacement(
                base_offset,
                detail_offset,
                self.dcn_weight,
                self.kernel_size,
            ),
        }
