"""Default paper configuration for DIAF-Net."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class DIAFConfig:
    # Data
    data_root: str = "datasets/BraTS2020"
    dataset: str = "brats2020"  # brats2020 or fastmri
    scale: int = 4
    channels: int = 3
    train_split: str = "train"
    val_split: str = "val"
    val_condition: str = "clean"
    train_misalignment: bool = True
    misalignment_probability: float = 0.5
    max_rotation_deg: float = 5.0
    max_translation_px: float = 5.0

    # DIAF-Net
    dims: Tuple[int, int, int, int] = (64, 128, 192, 288)
    encoder_depths: Tuple[int, int, int, int] = (2, 2, 2, 2)
    decoder_depths: Tuple[int, int, int, int] = (2, 2, 2, 2)
    window_sizes: Tuple[int, int, int, int] = (8, 8, 8, 4)
    num_heads: Tuple[int, int, int, int] = (4, 8, 8, 12)
    tokens_per_window: Tuple[int, int, int, int] = (1, 1, 2, 2)
    max_base_offsets: Tuple[float, float, float, float] = (4.0, 4.0, 3.0, 2.0)
    max_detail_offsets: Tuple[float, float, float, float] = (2.0, 2.0, 1.5, 1.0)
    backbone_mlp_ratio: float = 2.0
    attention_mlp_ratio: float = 2.0
    drop_path_rate: float = 0.0
    routing_temperature: float = 0.7
    sfdm_std_floor: float = 0.1
    sfdm_clip: float = 3.0
    router_beta_max: float = 3.0
    base_global_mix: float = 0.25
    global_mix_scale: float = 0.10
    global_mix_min: float = 0.10
    global_mix_max: float = 0.40
    use_global_residual: bool = True

    # Optimization
    epochs: int = 200
    batch_size: int = 2
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    min_learning_rate: float = 1e-5
    lfcl_weight: float = 1e-4
    uedc_weight: float = 0.05
    uedc_max_translation_px: int = 5
    uedc_smooth_l1_beta: float = 0.5
    uedc_ignore_border: bool = True
    uedc_flow_sign: float = 1.0
    seed: int = 41
    amp: bool = False
    device: str = "cuda"

    # Logging/checkpoints
    output_dir: str = "results/diaf_net"
    save_every: int = 10

    def model_kwargs(self) -> Dict[str, Any]:
        return {
            "in_channels": self.channels,
            "out_channels": self.channels,
            "dims": self.dims,
            "encoder_depths": self.encoder_depths,
            "decoder_depths": self.decoder_depths,
            "window_sizes": self.window_sizes,
            "num_heads": self.num_heads,
            "tokens_per_window": self.tokens_per_window,
            "max_base_offsets": self.max_base_offsets,
            "max_detail_offsets": self.max_detail_offsets,
            "backbone_mlp_ratio": self.backbone_mlp_ratio,
            "attention_mlp_ratio": self.attention_mlp_ratio,
            "drop_path_rate": self.drop_path_rate,
            "routing_temperature": self.routing_temperature,
            "sfdm_std_floor": self.sfdm_std_floor,
            "sfdm_clip": self.sfdm_clip,
            "router_beta_max": self.router_beta_max,
            "base_global_mix": self.base_global_mix,
            "global_mix_scale": self.global_mix_scale,
            "global_mix_min": self.global_mix_min,
            "global_mix_max": self.global_mix_max,
            "use_global_residual": self.use_global_residual,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
