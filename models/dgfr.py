"""Dispersion-Guided Fusion Refinement (DGFR)."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep_prob) / keep_prob


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for BCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def _pad_to_window(x: Tensor, window_size: int) -> Tuple[Tensor, int, int]:
    if x.ndim != 4:
        raise ValueError(f"x must be BCHW, got {tuple(x.shape)}")
    _, _, height, width = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, height, width


def _window_partition(x: Tensor, window_size: int) -> Tuple[Tensor, int, int]:
    batch, channels, height, width = x.shape
    if height % window_size or width % window_size:
        raise ValueError("spatial dimensions must be divisible by window_size")
    num_h, num_w = height // window_size, width // window_size
    windows = (
        x.view(batch, channels, num_h, window_size, num_w, window_size)
        .permute(0, 2, 4, 3, 5, 1)
        .contiguous()
        .view(batch, num_h * num_w, window_size * window_size, channels)
    )
    return windows, num_h, num_w


def _window_reverse(
    windows: Tensor,
    window_size: int,
    num_h: int,
    num_w: int,
) -> Tensor:
    batch, num_windows, num_tokens, channels = windows.shape
    if num_windows != num_h * num_w or num_tokens != window_size * window_size:
        raise ValueError("invalid window tensor shape")
    return (
        windows.view(batch, num_h, num_w, window_size, window_size, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .contiguous()
        .view(batch, channels, num_h * window_size, num_w * window_size)
    )


def _relative_position_index(window_size: int) -> Tensor:
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing="ij",
        )
    ).flatten(1)
    relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
    relative[:, :, 0] += window_size - 1
    relative[:, :, 1] += window_size - 1
    relative[:, :, 0] *= 2 * window_size - 1
    return relative.sum(dim=-1)


def _calibrate_sfdm(
    sfdm_scores: Tensor,
    std_floor: float,
    clip_value: float,
) -> Tensor:
    mean = sfdm_scores.mean(dim=-1, keepdim=True)
    variance = (sfdm_scores - mean).square().mean(dim=-1, keepdim=True)
    relative = (sfdm_scores - mean) / (torch.sqrt(variance + 1e-8) + std_floor)
    if clip_value > 0:
        relative = relative.clamp(-clip_value, clip_value)
    return relative


class InformativeTokenRouter(nn.Module):
    """Aggregate representative tokens with content and relative-SFDM routing."""

    def __init__(
        self,
        channels: int,
        tokens_per_window: int,
        route_dim: int | None = None,
        temperature: float = 0.7,
        sfdm_std_floor: float = 0.1,
        sfdm_clip: float = 3.0,
        beta_max: float = 3.0,
    ) -> None:
        super().__init__()
        if tokens_per_window <= 0:
            raise ValueError("tokens_per_window must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if beta_max <= 0:
            raise ValueError("beta_max must be positive")

        self.channels = int(channels)
        self.tokens_per_window = int(tokens_per_window)
        self.route_dim = int(route_dim or max(channels // 4, 8))
        self.temperature = float(temperature)
        self.sfdm_std_floor = float(sfdm_std_floor)
        self.sfdm_clip = float(sfdm_clip)
        self.beta_max = float(beta_max)

        self.router_queries = nn.Parameter(
            torch.empty(self.tokens_per_window, self.route_dim)
        )
        self.route_key = nn.Linear(channels, self.route_dim, bias=False)
        self.route_value = nn.Linear(channels, channels, bias=False)

        beta_init = torch.zeros(self.tokens_per_window)
        beta_init[0] = min(1.0, 0.95 * self.beta_max)
        normalized = (beta_init / self.beta_max).clamp(-0.999, 0.999)
        self.router_beta_raw = nn.Parameter(torch.atanh(normalized))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.router_queries, std=0.02)
        nn.init.xavier_uniform_(self.route_key.weight)
        nn.init.xavier_uniform_(self.route_value.weight)

    def forward(self, feature_windows: Tensor, sfdm_windows: Tensor) -> Tensor:
        if feature_windows.ndim != 4 or sfdm_windows.ndim != 4:
            raise ValueError("feature_windows and sfdm_windows must be [B,nW,N,C]")
        if feature_windows.shape[:3] != sfdm_windows.shape[:3] or sfdm_windows.shape[-1] != 1:
            raise ValueError("feature and SFDM window dimensions do not match")

        route_keys = self.route_key(feature_windows)
        content_logits = torch.einsum(
            "tr,bwnr->bwtn",
            self.router_queries,
            route_keys,
        ) * (self.route_dim ** -0.5)
        content_mean = content_logits.mean(dim=-1, keepdim=True)
        content_var = (content_logits - content_mean).square().mean(dim=-1, keepdim=True)
        content_logits = (content_logits - content_mean) / torch.sqrt(content_var + 1e-6)

        relative_sfdm = _calibrate_sfdm(
            sfdm_windows.squeeze(-1),
            self.sfdm_std_floor,
            self.sfdm_clip,
        )
        beta = self.beta_max * torch.tanh(self.router_beta_raw)
        routing_logits = (
            content_logits + beta.view(1, 1, -1, 1) * relative_sfdm.unsqueeze(2)
        ) / self.temperature
        routing_weights = F.softmax(routing_logits, dim=-1)
        route_values = self.route_value(feature_windows)
        return torch.einsum("bwtn,bwnc->bwtc", routing_weights, route_values)


class DispersionGuidedLocalGlobalAttention(nn.Module):
    """Window self-attention and routed sparse global cross-attention."""

    def __init__(
        self,
        channels: int,
        window_size: int,
        num_heads: int,
        tokens_per_window: int,
        route_dim: int | None = None,
        routing_temperature: float = 0.7,
        sfdm_std_floor: float = 0.1,
        sfdm_clip: float = 3.0,
        router_beta_max: float = 3.0,
        base_global_mix: float = 0.25,
        global_mix_scale: float = 0.10,
        global_mix_min: float = 0.10,
        global_mix_max: float = 0.40,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if not 0.0 <= base_global_mix <= 1.0:
            raise ValueError("base_global_mix must be in [0,1]")
        if not 0.0 <= global_mix_min <= global_mix_max <= 1.0:
            raise ValueError("global mix bounds must satisfy 0 <= min <= max <= 1")

        self.channels = int(channels)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.base_global_mix = float(base_global_mix)
        self.global_mix_scale = float(global_mix_scale)
        self.global_mix_min = float(global_mix_min)
        self.global_mix_max = float(global_mix_max)
        self.sfdm_std_floor = float(sfdm_std_floor)
        self.sfdm_clip = float(sfdm_clip)

        self.local_qkv = nn.Linear(channels, 3 * channels)
        self.token_router = InformativeTokenRouter(
            channels=channels,
            tokens_per_window=tokens_per_window,
            route_dim=route_dim,
            temperature=routing_temperature,
            sfdm_std_floor=sfdm_std_floor,
            sfdm_clip=sfdm_clip,
            beta_max=router_beta_max,
        )
        self.global_kv = nn.Linear(channels, 2 * channels)
        self.output_proj = nn.Linear(channels, channels)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        num_relative_positions = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, self.num_heads)
        )
        self.register_buffer(
            "relative_position_index",
            _relative_position_index(self.window_size),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.local_qkv, self.global_kv, self.output_proj):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor, sfdm: Tensor) -> Tensor:
        if x.ndim != 4 or sfdm.ndim != 4 or sfdm.shape[1] != 1:
            raise ValueError("x must be BCHW and sfdm must be [B,1,H,W]")
        if sfdm.shape[-2:] != x.shape[-2:]:
            sfdm = F.interpolate(
                sfdm,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x_pad, original_h, original_w = _pad_to_window(x, self.window_size)
        sfdm_pad, _, _ = _pad_to_window(sfdm, self.window_size)
        feature_windows, num_h, num_w = _window_partition(x_pad, self.window_size)
        sfdm_windows, _, _ = _window_partition(sfdm_pad, self.window_size)
        batch, num_windows, num_tokens, channels = feature_windows.shape

        qkv = self.local_qkv(feature_windows).view(
            batch,
            num_windows,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, local_key, local_value = qkv.permute(3, 0, 1, 4, 2, 5).unbind(0)
        local_logits = torch.matmul(query, local_key.transpose(-2, -1)) * self.scale
        relative_bias = (
            self.relative_position_bias_table[
                self.relative_position_index.reshape(-1)
            ]
            .view(num_tokens, num_tokens, self.num_heads)
            .permute(2, 0, 1)
            .contiguous()
        )
        local_attention = self.attn_drop(
            F.softmax(local_logits + relative_bias[None, None], dim=-1)
        )
        local_output = torch.matmul(local_attention, local_value)

        routed_tokens = self.token_router(feature_windows, sfdm_windows)
        global_library = routed_tokens.reshape(batch, -1, channels)
        num_global_tokens = global_library.shape[1]
        global_kv = self.global_kv(global_library).view(
            batch,
            num_global_tokens,
            2,
            self.num_heads,
            self.head_dim,
        )
        global_key, global_value = global_kv.permute(2, 0, 3, 1, 4).unbind(0)
        global_logits = torch.einsum(
            "bwhnd,bhld->bwhnl",
            query,
            global_key,
        ) * self.scale
        global_attention = self.attn_drop(F.softmax(global_logits, dim=-1))
        global_output = torch.einsum(
            "bwhnl,bhld->bwhnd",
            global_attention,
            global_value,
        )

        relative_sfdm = _calibrate_sfdm(
            sfdm_windows.squeeze(-1),
            self.sfdm_std_floor,
            self.sfdm_clip,
        )
        # The mixing map is a forward modulation signal; router gradients still
        # train SFDE through the routing logits.
        alpha = self.base_global_mix + self.global_mix_scale * torch.tanh(
            relative_sfdm.detach()
        )
        alpha = alpha.clamp(self.global_mix_min, self.global_mix_max)
        alpha = alpha.unsqueeze(2).unsqueeze(-1)

        mixed = (1.0 - alpha) * local_output + alpha * global_output
        mixed = mixed.permute(0, 1, 3, 2, 4).contiguous().view(
            batch,
            num_windows,
            num_tokens,
            channels,
        )
        mixed = self.proj_drop(self.output_proj(mixed))
        output = _window_reverse(mixed, self.window_size, num_h, num_w)
        return output[:, :, :original_h, :original_w]


class ConvFFN(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 2.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Dropout(drop),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DispersionGuidedRefinementBlock(nn.Module):
    """Residual local-global attention followed by a convolutional FFN."""

    def __init__(
        self,
        channels: int,
        window_size: int,
        num_heads: int,
        tokens_per_window: int,
        route_dim: int | None = None,
        routing_temperature: float = 0.7,
        sfdm_std_floor: float = 0.1,
        sfdm_clip: float = 3.0,
        router_beta_max: float = 3.0,
        base_global_mix: float = 0.25,
        global_mix_scale: float = 0.10,
        global_mix_min: float = 0.10,
        global_mix_max: float = 0.40,
        mlp_ratio: float = 2.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attn = DispersionGuidedLocalGlobalAttention(
            channels=channels,
            window_size=window_size,
            num_heads=num_heads,
            tokens_per_window=tokens_per_window,
            route_dim=route_dim,
            routing_temperature=routing_temperature,
            sfdm_std_floor=sfdm_std_floor,
            sfdm_clip=sfdm_clip,
            router_beta_max=router_beta_max,
            base_global_mix=base_global_mix,
            global_mix_scale=global_mix_scale,
            global_mix_min=global_mix_min,
            global_mix_max=global_mix_max,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = LayerNorm2d(channels)
        self.ffn = ConvFFN(channels, mlp_ratio=mlp_ratio, drop=ffn_drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: Tensor, sfdm: Tensor) -> Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x), sfdm))
        return x + self.drop_path2(self.ffn(self.norm2(x)))


def _aggregation_hidden_width(channels: int) -> int:
    """Return the hidden width used by the validated DIAF-Net implementation."""
    match_channels = max(channels // 4, 8)
    target_params = (
        2 * channels * match_channels
        + 2 * match_channels
        + 2 * channels * channels
        + 2 * channels
    )

    def params(hidden: int) -> int:
        return 3 * channels * hidden + 9 * hidden * hidden + 2 * hidden + channels

    candidates = range(4, max(96, channels * 6) + 1)
    return min(candidates, key=lambda hidden: abs(params(hidden) - target_params))


class TargetAnchoredAggregation(nn.Module):
    """Fuse aligned reference information as a residual around the target feature."""

    def __init__(self, channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_channels or _aggregation_hidden_width(channels))
        self.body = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.body[:-1]:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, target_feature: Tensor, aligned_reference: Tensor) -> Tensor:
        if target_feature.shape != aligned_reference.shape:
            raise ValueError("target_feature and aligned_reference must have the same shape")
        residual = self.body(torch.cat((target_feature, aligned_reference), dim=1))
        return target_feature + residual


class DispersionGuidedFusionRefinement(nn.Module):
    """Target-anchored aggregation followed by SFDM-guided refinement."""

    def __init__(
        self,
        channels: int,
        window_size: int,
        num_heads: int,
        tokens_per_window: int,
        route_dim: int | None = None,
        routing_temperature: float = 0.7,
        sfdm_std_floor: float = 0.1,
        sfdm_clip: float = 3.0,
        router_beta_max: float = 3.0,
        base_global_mix: float = 0.25,
        global_mix_scale: float = 0.10,
        global_mix_min: float = 0.10,
        global_mix_max: float = 0.40,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.aggregation = TargetAnchoredAggregation(channels)
        self.refinement = DispersionGuidedRefinementBlock(
            channels=channels,
            window_size=window_size,
            num_heads=num_heads,
            tokens_per_window=tokens_per_window,
            route_dim=route_dim,
            routing_temperature=routing_temperature,
            sfdm_std_floor=sfdm_std_floor,
            sfdm_clip=sfdm_clip,
            router_beta_max=router_beta_max,
            base_global_mix=base_global_mix,
            global_mix_scale=global_mix_scale,
            global_mix_min=global_mix_min,
            global_mix_max=global_mix_max,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
        )

    def forward(
        self,
        target_feature: Tensor,
        aligned_reference: Tensor,
        sfdm: Tensor,
    ) -> Tensor:
        fused = self.aggregation(target_feature, aligned_reference)
        return self.refinement(fused, sfdm)
