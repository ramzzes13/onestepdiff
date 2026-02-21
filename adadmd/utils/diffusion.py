"""Diffusion utilities: noise schedules, forward diffusion, etc."""

import torch
import math


def linear_beta_schedule(num_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos((steps / num_timesteps + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999).float()


class DiffusionSchedule:
    """Manages noise schedule parameters for DDPM."""

    def __init__(self, num_timesteps: int = 1000, schedule_type: str = "linear",
                 beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps

        if schedule_type == "linear":
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        # For the sigma_t / alpha_t weighting in DMD
        self.sigma_t = self.sqrt_one_minus_alphas_cumprod
        self.alpha_t = self.sqrt_alphas_cumprod

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sigma_t = self.sigma_t.to(device)
        self.alpha_t = self.alpha_t.to(device)
        return self

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I)"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def sample_timesteps(self, batch_size: int, t_min: float = 0.02, t_max: float = 0.98,
                         device: torch.device = None) -> torch.Tensor:
        """Sample uniform timesteps in [t_min*T, t_max*T]."""
        low = int(t_min * self.num_timesteps)
        high = int(t_max * self.num_timesteps)
        return torch.randint(low, high, (batch_size,), device=device)
