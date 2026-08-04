"""DIAF-Net for multi-contrast MRI super-resolution."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .dgca import DispersionGuidedCoarseToFineAlignment
from .dgfr import DispersionGuidedFusionRefinement
from .sfde import SpatialFeatureDispersionEstimator


def _as_tuple(value, length: int, name: str) -> Tuple:
    if isinstance(value, (tuple, list)):
        if len(value) != length:
            raise ValueError(f"{name} must contain {length} values")
        return tuple(value)
    return (value,) * length


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0,1)")
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep_prob) / keep_prob


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class LiteConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 2.0,
        layer_scale_init: float = 1e-6,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = max(channels, int(round(channels * mlp_ratio)))
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.norm = LayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.activation = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.pwconv2(self.activation(self.pwconv1(self.norm(self.dwconv(x)))))
        x = x * self.gamma[:, None, None]
        return residual + self.drop_path(x)


class LiteConvNeXtStage(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        mlp_ratio: float,
        drop_paths=0.0,
    ) -> None:
        super().__init__()
        rates = _as_tuple(drop_paths, depth, "drop_paths")
        self.blocks = nn.Sequential(
            *[
                LiteConvNeXtBlock(
                    channels=channels,
                    mlp_ratio=mlp_ratio,
                    drop_path=float(rates[index]),
                )
                for index in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class ModalityStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            LayerNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stem(x)


class ConvDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class SharedFourStageEncoder(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        depths: Sequence[int],
        mlp_ratio: float,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("dims and depths must each contain four values")
        rates = torch.linspace(0.0, drop_path_rate, int(sum(depths))).tolist()
        cursor = 0
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for level in range(4):
            depth = int(depths[level])
            self.stages.append(
                LiteConvNeXtStage(
                    channels=int(dims[level]),
                    depth=depth,
                    mlp_ratio=mlp_ratio,
                    drop_paths=rates[cursor : cursor + depth],
                )
            )
            cursor += depth
            if level < 3:
                self.downsamples.append(
                    ConvDownsample(int(dims[level]), int(dims[level + 1]))
                )

    def forward(self, x: Tensor) -> List[Tensor]:
        features: List[Tensor] = []
        for level, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if level < 3:
                x = self.downsamples[level](x)
        return features


class DIAFInteraction(nn.Module):
    """One scale of DGCA followed by DGFR."""

    def __init__(
        self,
        channels: int,
        window_size: int,
        num_heads: int,
        tokens_per_window: int,
        max_base_offset: float,
        max_detail_offset: float,
        attention_mlp_ratio: float,
        routing_temperature: float,
        sfdm_std_floor: float,
        sfdm_clip: float,
        router_beta_max: float,
        base_global_mix: float,
        global_mix_scale: float,
        global_mix_min: float,
        global_mix_max: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.dgca = DispersionGuidedCoarseToFineAlignment(
            channels=channels,
            hidden_channels=max(channels // 2, 16),
            kernel_size=3,
            max_base_offset=max_base_offset,
            max_detail_offset=max_detail_offset,
        )
        self.dgfr = DispersionGuidedFusionRefinement(
            channels=channels,
            window_size=window_size,
            num_heads=num_heads,
            tokens_per_window=tokens_per_window,
            route_dim=max(channels // 4, 8),
            routing_temperature=routing_temperature,
            sfdm_std_floor=sfdm_std_floor,
            sfdm_clip=sfdm_clip,
            router_beta_max=router_beta_max,
            base_global_mix=base_global_mix,
            global_mix_scale=global_mix_scale,
            global_mix_min=global_mix_min,
            global_mix_max=global_mix_max,
            mlp_ratio=attention_mlp_ratio,
            drop_path=drop_path,
        )

    def forward(
        self,
        target_feature: Tensor,
        reference_feature: Tensor,
        sfdm: Tensor,
        return_alignment_aux: bool = False,
    ):
        if return_alignment_aux:
            aligned_reference, alignment_aux = self.dgca(
                target_feature,
                reference_feature,
                sfdm,
                return_aux=True,
            )
            output = self.dgfr(target_feature, aligned_reference, sfdm)
            return output, alignment_aux

        aligned_reference = self.dgca(target_feature, reference_feature, sfdm)
        return self.dgfr(target_feature, aligned_reference, sfdm)

    def alignment_aux(
        self,
        target_feature: Tensor,
        reference_feature: Tensor,
        sfdm: Tensor,
    ) -> Dict[str, Tensor]:
        _, auxiliary = self.dgca(
            target_feature,
            reference_feature,
            sfdm,
            return_aux=True,
        )
        return auxiliary


class ConvNeXtUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class DeepestDecoderStage(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        mlp_ratio: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.convnext_stage = LiteConvNeXtStage(
            channels=channels,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_paths=drop_path,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.convnext_stage(x)


class UpConcatDecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        depth: int,
        mlp_ratio: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.upsample = ConvNeXtUpsample(in_channels, out_channels)
        self.skip_proj = (
            nn.Identity()
            if skip_channels == out_channels
            else nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        )
        self.concat_fuse = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=1),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )
        self.convnext_stage = LiteConvNeXtStage(
            channels=out_channels,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_paths=drop_path,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upsample(x)
        skip = self.skip_proj(skip)
        return self.convnext_stage(self.concat_fuse(torch.cat((x, skip), dim=1)))


class FourStageDecoder(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        depths: Sequence[int],
        mlp_ratio: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = [int(value) for value in dims]
        d1, d2, d3, d4 = [int(value) for value in depths]
        self.decoder4 = DeepestDecoderStage(c4, d4, mlp_ratio, drop_path)
        self.decoder3 = UpConcatDecoderStage(c4, c3, c3, d3, mlp_ratio, drop_path)
        self.decoder2 = UpConcatDecoderStage(c3, c2, c2, d2, mlp_ratio, drop_path)
        self.decoder1 = UpConcatDecoderStage(c2, c1, c1, d1, mlp_ratio, drop_path)


class DIAFNet(nn.Module):
    """Dispersion-Informed Alignment and Fusion Network.

    Args:
        target_lr: low-resolution target MRI, [B, C, h, w].
        reference_hr: potentially misaligned HR reference MRI, [B, C, H, W].

    The target is bilinearly resized to the reference resolution before encoding.
    SFDE is applied once to the highest-resolution target feature; its SFDM is
    resized into a four-scale pyramid that guides DGCA and DGFR at every scale.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dims: Sequence[int] = (64, 128, 192, 288),
        encoder_depths: Sequence[int] = (2, 2, 2, 2),
        decoder_depths: Sequence[int] = (2, 2, 2, 2),
        window_sizes: Sequence[int] = (8, 8, 8, 4),
        num_heads: Sequence[int] = (4, 8, 8, 12),
        tokens_per_window: Sequence[int] = (1, 1, 2, 2),
        max_base_offsets: Sequence[float] = (4.0, 4.0, 3.0, 2.0),
        max_detail_offsets: Sequence[float] = (2.0, 2.0, 1.5, 1.0),
        backbone_mlp_ratio: float = 2.0,
        attention_mlp_ratio: float = 2.0,
        drop_path_rate: float = 0.0,
        routing_temperature: float = 0.7,
        sfdm_std_floor: float = 0.1,
        sfdm_clip: float = 3.0,
        router_beta_max: float = 3.0,
        base_global_mix: float = 0.25,
        global_mix_scale: float = 0.10,
        global_mix_min: float = 0.10,
        global_mix_max: float = 0.40,
        use_global_residual: bool = True,
    ) -> None:
        super().__init__()
        for name, values in (
            ("dims", dims),
            ("encoder_depths", encoder_depths),
            ("decoder_depths", decoder_depths),
            ("window_sizes", window_sizes),
            ("num_heads", num_heads),
            ("tokens_per_window", tokens_per_window),
            ("max_base_offsets", max_base_offsets),
            ("max_detail_offsets", max_detail_offsets),
        ):
            if len(values) != 4:
                raise ValueError(f"{name} must contain four values")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dims = tuple(int(value) for value in dims)
        self.use_global_residual = bool(use_global_residual)

        self.target_stem = ModalityStem(in_channels, self.dims[0])
        self.reference_stem = ModalityStem(in_channels, self.dims[0])
        self.shared_encoder = SharedFourStageEncoder(
            dims=self.dims,
            depths=encoder_depths,
            mlp_ratio=backbone_mlp_ratio,
            drop_path_rate=drop_path_rate,
        )
        self.sfde = SpatialFeatureDispersionEstimator(self.dims[0])

        self.interactions = nn.ModuleList(
            [
                DIAFInteraction(
                    channels=self.dims[level],
                    window_size=int(window_sizes[level]),
                    num_heads=int(num_heads[level]),
                    tokens_per_window=int(tokens_per_window[level]),
                    max_base_offset=float(max_base_offsets[level]),
                    max_detail_offset=float(max_detail_offsets[level]),
                    attention_mlp_ratio=attention_mlp_ratio,
                    routing_temperature=routing_temperature,
                    sfdm_std_floor=sfdm_std_floor,
                    sfdm_clip=sfdm_clip,
                    router_beta_max=router_beta_max,
                    base_global_mix=base_global_mix,
                    global_mix_scale=global_mix_scale,
                    global_mix_min=global_mix_min,
                    global_mix_max=global_mix_max,
                    drop_path=drop_path_rate,
                )
                for level in range(4)
            ]
        )

        self.decoder = FourStageDecoder(
            dims=self.dims,
            depths=decoder_depths,
            mlp_ratio=backbone_mlp_ratio,
            drop_path=drop_path_rate,
        )
        self.output_head = nn.Sequential(
            nn.Conv2d(self.dims[0], self.dims[0], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.dims[0], out_channels, kernel_size=3, padding=1),
        )
        self.residual_proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def _encode_target(self, target: Tensor) -> List[Tensor]:
        return self.shared_encoder(self.target_stem(target))

    def _encode_reference(self, reference: Tensor) -> List[Tensor]:
        return self.shared_encoder(self.reference_stem(reference))

    def _prepare_features(
        self,
        target_lr: Tensor,
        reference_hr: Tensor,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor], Dict[str, Tensor], List[Tensor]]:
        if target_lr.ndim != 4 or reference_hr.ndim != 4:
            raise ValueError("target_lr and reference_hr must be BCHW tensors")
        if target_lr.shape[0] != reference_hr.shape[0]:
            raise ValueError("target_lr and reference_hr must share the batch size")

        target_up = F.interpolate(
            target_lr,
            size=reference_hr.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        target_features = self._encode_target(target_up)
        reference_features = self._encode_reference(reference_hr)
        sfde_output = self.sfde(target_features[0])
        sfdm = sfde_output["sfdm"]
        sfdm_pyramid = [sfdm]
        for feature in target_features[1:]:
            sfdm_pyramid.append(
                F.interpolate(
                    sfdm,
                    size=feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return (
            target_up,
            target_features,
            reference_features,
            sfde_output,
            sfdm_pyramid,
        )

    def forward(
        self,
        target_lr: Tensor,
        reference_hr: Tensor,
        return_aux: bool = False,
        alignment_aux_only: bool = False,
    ):
        (
            target_up,
            target_features,
            reference_features,
            sfde_output,
            sfdm_pyramid,
        ) = self._prepare_features(target_lr, reference_hr)

        if alignment_aux_only:
            return {
                "effective_offsets": tuple(
                    self.interactions[level].alignment_aux(
                        target_features[level],
                        reference_features[level],
                        sfdm_pyramid[level],
                    )["effective_offset"]
                    for level in range(4)
                )
            }

        enhanced: List[Tensor] = []
        effective_offsets: List[Tensor] = []
        for level in range(4):
            if return_aux:
                feature, alignment_aux = self.interactions[level](
                    target_features[level],
                    reference_features[level],
                    sfdm_pyramid[level],
                    return_alignment_aux=True,
                )
                enhanced.append(feature)
                effective_offsets.append(alignment_aux["effective_offset"])
            else:
                enhanced.append(
                    self.interactions[level](
                        target_features[level],
                        reference_features[level],
                        sfdm_pyramid[level],
                    )
                )

        x = self.decoder.decoder4(enhanced[3])
        x = self.decoder.decoder3(x, enhanced[2])
        x = self.decoder.decoder2(x, enhanced[1])
        x = self.decoder.decoder1(x, enhanced[0])
        residual = self.output_head(x)
        output = (
            self.residual_proj(target_up) + residual
            if self.use_global_residual
            else residual
        )

        if not return_aux:
            return output
        return output, {
            "sfde": sfde_output,
            "sfdm": sfde_output["sfdm"],
            "sfdm_pyramid": tuple(sfdm_pyramid),
            "effective_offsets": tuple(effective_offsets),
        }



def build_diaf_net(**kwargs) -> DIAFNet:
    """Convenience factory used by training and evaluation scripts."""
    return DIAFNet(**kwargs)
