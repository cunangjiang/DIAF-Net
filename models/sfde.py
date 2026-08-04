"""Spatial Feature Dispersion Estimator (SFDE)."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _standard_gaussian_cdf(x: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def normalize_sfdm(scale: Tensor, eps: float = 1e-6) -> Tensor:
    """Convert a positive channel-wise scale tensor into a [0, 1] SFDM."""
    if scale.ndim != 4:
        raise ValueError(f"scale must be BCHW, got {tuple(scale.shape)}")
    scale_map = scale.mean(dim=1, keepdim=True)
    flat = scale_map.flatten(2)
    minimum = flat.amin(dim=-1, keepdim=True).unsqueeze(-1)
    maximum = flat.amax(dim=-1, keepdim=True).unsqueeze(-1)
    return ((scale_map - minimum) / (maximum - minimum + eps)).clamp_(0.0, 1.0)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class GDNLite(nn.Module):
    """Lightweight channel-wise generalized divisive normalization."""

    def __init__(
        self,
        channels: int,
        beta_init: float = 1.0,
        gamma_init: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.beta = nn.Parameter(
            torch.full((channels,), _inverse_softplus(beta_init))
        )
        self.gamma = nn.Parameter(
            torch.full((channels,), _inverse_softplus(gamma_init))
        )

    def forward(self, x: Tensor) -> Tensor:
        beta = F.softplus(self.beta).view(1, -1, 1, 1) + self.eps
        gamma = F.softplus(self.gamma).view(1, -1, 1, 1)
        return x / torch.sqrt(beta + gamma * x.square() + self.eps)


class SpatialFeatureDispersionEstimator(nn.Module):
    """Estimate a Gaussian feature model and derive the SFDM from its scale.

    During training, additive uniform noise approximates quantization. During
    evaluation, rounding is used. The returned likelihood is consumed by LFCL.
    """

    def __init__(
        self,
        channels: int,
        min_scale: float = 1e-6,
        likelihood_eps: float = 1e-9,
    ) -> None:
        super().__init__()
        self.min_scale = float(min_scale)
        self.likelihood_eps = float(likelihood_eps)

        self.mean_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            GDNLite(channels),
        )
        self.scale_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            GDNLite(channels),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=0.1,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _quantize(self, feature: Tensor) -> Tensor:
        if self.training:
            return feature + torch.empty_like(feature).uniform_(-0.5, 0.5)
        return torch.round(feature)

    def _likelihood(self, quantized: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        upper = (quantized + 0.5 - mean) / scale
        lower = (quantized - 0.5 - mean) / scale
        likelihood = _standard_gaussian_cdf(upper) - _standard_gaussian_cdf(lower)
        return likelihood.clamp_min(self.likelihood_eps)

    def forward(self, target_feature: Tensor) -> Dict[str, Tensor]:
        if target_feature.ndim != 4:
            raise ValueError(
                f"target_feature must be BCHW, got {tuple(target_feature.shape)}"
            )

        mean = self.mean_branch(target_feature)
        scale = F.softplus(self.scale_branch(target_feature)) + self.min_scale
        sfdm = normalize_sfdm(scale)
        quantized = self._quantize(target_feature)
        likelihood = self._likelihood(quantized, mean, scale)

        return {
            "mean": mean,
            "scale": scale,
            "sfdm": sfdm,
            "quantized": quantized,
            "likelihood": likelihood,
            "bits": -torch.log2(likelihood),
        }
