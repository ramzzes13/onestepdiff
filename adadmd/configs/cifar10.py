"""Configuration for CIFAR-10 experiments."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CIFAR10Config:
    # Data
    dataset: str = "cifar10"
    image_size: int = 32
    in_channels: int = 3
    num_classes: int = 10

    # Base UNet
    base_channels: int = 64
    num_res_blocks: int = 2

    # Diffusion
    num_timesteps: int = 1000
    schedule_type: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    t_min: float = 0.02
    t_max: float = 0.98

    # LoRA fake score model
    lora_rank_init: int = 4
    lora_rank_max: int = 32
    lora_alpha: float = 1.0
    lora_target_modules: Tuple[str, ...] = ('q', 'k', 'v', 'out')
    rank_warm_interval: int = 5000

    # Density ratio network
    dr_base_channels: int = 32

    # Training
    batch_size: int = 32
    lr_generator: float = 5e-5
    lr_lora: float = 1e-4
    lr_density_ratio: float = 1e-4
    weight_decay: float = 0.0
    num_iterations: int = 50000
    gradient_checkpointing: bool = False
    mixed_precision: bool = False  # Disable for small models (GradScaler issues)

    # Loss weights
    lambda_lpips: float = 0.1
    lambda_dr: float = 0.5
    dr_margin: float = 1.0
    use_lpips: bool = False  # Disable for CIFAR-10 (use L1 instead)

    # EMA
    ema_decay: float = 0.9999
    ema_dr_decay: float = 0.99

    # Logging
    log_interval: int = 100
    save_interval: int = 5000
    eval_interval: int = 5000
    num_eval_samples: int = 5000
    output_dir: str = "outputs/cifar10"

    # Device
    device: str = "cuda"
    gpu_id: int = 0


@dataclass
class ImageNet64Config(CIFAR10Config):
    dataset: str = "imagenet64"
    image_size: int = 64
    num_classes: int = 1000
    base_channels: int = 128

    lora_rank_init: int = 4
    lora_rank_max: int = 64
    rank_warm_interval: int = 10000

    dr_base_channels: int = 32

    batch_size: int = 16
    lr_generator: float = 2e-6
    lr_lora: float = 5e-5
    num_iterations: int = 100000
    gradient_checkpointing: bool = True

    num_eval_samples: int = 10000
    output_dir: str = "outputs/imagenet64"
