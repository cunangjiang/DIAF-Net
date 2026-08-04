"""Training objectives used by DIAF-Net."""

from __future__ import annotations

import math
from typing import Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LikelihoodBasedFeatureCodingLoss(nn.Module):
    """LFCL: expected code length implied by the SFDE likelihood model."""

    def __init__(self, reduction: str = "mean", eps: float = 1e-9) -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.reduction = reduction
        self.eps = float(eps)

    def forward(self, sfde_output: Mapping[str, Tensor] | Tensor) -> Tensor:
        if torch.is_tensor(sfde_output):
            bits = -torch.log2(sfde_output.clamp_min(self.eps))
        else:
            bits = sfde_output.get("bits")
            if bits is None:
                likelihood = sfde_output.get("likelihood")
                if likelihood is None:
                    raise KeyError("SFDE output must contain 'bits' or 'likelihood'")
                bits = -torch.log2(likelihood.clamp_min(self.eps))
        return bits.mean() if self.reduction == "mean" else bits.sum()


def translate_reference_batch(reference: Tensor, dx: Tensor, dy: Tensor) -> Tensor:
    """Translate each BCHW reference by integer image-pixel offsets."""
    if reference.ndim != 4:
        raise ValueError("reference must be a BCHW tensor")
    batch, _, height, width = reference.shape
    dx = torch.as_tensor(dx, device=reference.device, dtype=reference.dtype).reshape(-1)
    dy = torch.as_tensor(dy, device=reference.device, dtype=reference.dtype).reshape(-1)
    if dx.numel() == 1 and batch > 1:
        dx = dx.expand(batch)
    if dy.numel() == 1 and batch > 1:
        dy = dy.expand(batch)
    if dx.numel() != batch or dy.numel() != batch:
        raise ValueError("dx and dy must contain one value per sample")

    theta = reference.new_zeros((batch, 2, 3))
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * dx / max(width - 1, 1)
    theta[:, 1, 2] = -2.0 * dy / max(height - 1, 1)
    grid = F.affine_grid(theta, reference.shape, align_corners=True)
    return F.grid_sample(
        reference,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )


def sample_nonzero_translation(
    reference: Tensor,
    max_translation_px: int,
) -> Tuple[Tensor, Tensor]:
    """Sample one non-zero integer translation per batch sample."""
    maximum = int(round(max_translation_px))
    if maximum <= 0:
        raise ValueError("max_translation_px must be positive")
    batch = int(reference.shape[0])
    dx = torch.randint(-maximum, maximum + 1, (batch,), device=reference.device)
    dy = torch.randint(-maximum, maximum + 1, (batch,), device=reference.device)
    zero = (dx == 0) & (dy == 0)
    if zero.any():
        replacements = torch.where(
            torch.rand(int(zero.sum()), device=reference.device) < 0.5,
            torch.ones(int(zero.sum()), device=reference.device, dtype=dx.dtype),
            -torch.ones(int(zero.sum()), device=reference.device, dtype=dx.dtype),
        )
        dx[zero] = replacements
    return dx, dy


def _valid_translation_mask(
    dx: Tensor,
    dy: Tensor,
    feature_size: Sequence[int],
    image_size: Sequence[int],
    dtype: torch.dtype,
) -> Tensor:
    feature_height, feature_width = [int(value) for value in feature_size]
    image_height, image_width = [int(value) for value in image_size]
    mask = torch.ones(
        (dx.numel(), 1, feature_height, feature_width),
        device=dx.device,
        dtype=dtype,
    )
    scale_x = feature_width / max(image_width, 1)
    scale_y = feature_height / max(image_height, 1)
    for index in range(dx.numel()):
        border_x = int(math.ceil(abs(float(dx[index])) * scale_x))
        border_y = int(math.ceil(abs(float(dy[index])) * scale_y))
        if border_x > 0:
            if dx[index] > 0:
                mask[index, :, :, feature_width - border_x :] = 0
            else:
                mask[index, :, :, :border_x] = 0
        if border_y > 0:
            if dy[index] > 0:
                mask[index, :, feature_height - border_y :, :] = 0
            else:
                mask[index, :, :border_y, :] = 0
    return mask


def _masked_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    beta: float,
) -> Tensor:
    loss_map = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=float(beta),
    )
    denominator = mask.sum() * prediction.shape[1]
    return (loss_map * mask).sum() / denominator.clamp_min(1.0)


class UnifiedEffectiveDisplacementConsistencyLoss(nn.Module):
    """Supervise effective DGCA displacements under known translations."""

    def __init__(
        self,
        smooth_l1_beta: float = 0.5,
        ignore_translation_border: bool = True,
        flow_sign: float = 1.0,
    ) -> None:
        super().__init__()
        if smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.ignore_translation_border = bool(ignore_translation_border)
        self.flow_sign = float(flow_sign)

    def forward(
        self,
        native_offsets: Sequence[Tensor],
        shifted_offsets: Sequence[Tensor],
        dx: Tensor,
        dy: Tensor,
        reference_size: Sequence[int],
        mode: str,
    ) -> Tensor:
        if len(native_offsets) != 4 or len(shifted_offsets) != 4:
            raise ValueError("UEDC expects four alignment levels")
        mode = mode.lower()
        if mode not in {"absolute", "relative"}:
            raise ValueError("mode must be 'absolute' or 'relative'")

        image_height, image_width = [int(value) for value in reference_size]
        level_losses = []
        for native, shifted in zip(native_offsets, shifted_offsets):
            if native.shape != shifted.shape:
                raise ValueError("native and shifted effective offsets must match")
            _, _, height, width = native.shape
            target = torch.zeros_like(native)
            target[:, 0] = (
                self.flow_sign
                * dx.to(device=native.device, dtype=native.dtype)
                * (width / max(image_width, 1))
            )[:, None, None]
            target[:, 1] = (
                self.flow_sign
                * dy.to(device=native.device, dtype=native.dtype)
                * (height / max(image_height, 1))
            )[:, None, None]

            if self.ignore_translation_border:
                mask = _valid_translation_mask(
                    dx.to(native.device),
                    dy.to(native.device),
                    (height, width),
                    (image_height, image_width),
                    native.dtype,
                )
            else:
                mask = native.new_ones((native.shape[0], 1, height, width))

            if mode == "absolute":
                native_loss = _masked_smooth_l1(
                    native,
                    torch.zeros_like(native),
                    torch.ones_like(mask),
                    self.smooth_l1_beta,
                )
                shifted_loss = _masked_smooth_l1(
                    shifted,
                    target,
                    mask,
                    self.smooth_l1_beta,
                )
                level_loss = 0.5 * (native_loss + shifted_loss)
            else:
                level_loss = _masked_smooth_l1(
                    shifted - native.detach(),
                    target,
                    mask,
                    self.smooth_l1_beta,
                )
            level_losses.append(level_loss)
        return torch.stack(level_losses).mean()


class DIAFLoss(nn.Module):
    """Combined reconstruction, feature-coding, and alignment supervision."""

    def __init__(
        self,
        lfcl_weight: float = 1e-4,
        uedc_weight: float = 0.05,
        uedc_smooth_l1_beta: float = 0.5,
        uedc_ignore_border: bool = True,
        uedc_flow_sign: float = 1.0,
    ) -> None:
        super().__init__()
        if lfcl_weight < 0 or uedc_weight < 0:
            raise ValueError("loss weights must be non-negative")
        self.lfcl_weight = float(lfcl_weight)
        self.uedc_weight = float(uedc_weight)
        self.lfcl = LikelihoodBasedFeatureCodingLoss(reduction="mean")
        self.uedc = UnifiedEffectiveDisplacementConsistencyLoss(
            smooth_l1_beta=uedc_smooth_l1_beta,
            ignore_translation_border=uedc_ignore_border,
            flow_sign=uedc_flow_sign,
        )

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        sfde_output: Mapping[str, Tensor],
        native_offsets: Sequence[Tensor],
        shifted_offsets: Sequence[Tensor],
        dx: Tensor,
        dy: Tensor,
        reference_size: Sequence[int],
        uedc_mode: str,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        reconstruction = F.l1_loss(prediction, target)
        coding = self.lfcl(sfde_output)
        displacement = self.uedc(
            native_offsets=native_offsets,
            shifted_offsets=shifted_offsets,
            dx=dx,
            dy=dy,
            reference_size=reference_size,
            mode=uedc_mode,
        )
        total = (
            reconstruction
            + self.lfcl_weight * coding
            + self.uedc_weight * displacement
        )
        return total, {
            "total": total.detach(),
            "l1": reconstruction.detach(),
            "lfcl": coding.detach(),
            "uedc": displacement.detach(),
        }


LFCL = LikelihoodBasedFeatureCodingLoss
