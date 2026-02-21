"""Density Ratio Network for estimating log p_fake(x_t) / p_real(x_t).

Lightweight network trained with NCE to classify real vs fake noised samples.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityRatioNetwork(nn.Module):
    """Small convolutional network for density ratio estimation.

    Takes noised images x_t and timestep t, outputs log density ratio.
    Architecture: ~2-5M params (1/4 the channels of the base model).
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 32, image_size: int = 32):
        super().__init__()
        time_dim = base_ch * 4

        # Time embedding
        self.time_embed = nn.Sequential(
            SinPosEmbed(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder (no skip connections needed - just extract features)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.GroupNorm(min(16, base_ch * 2), base_ch * 2),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.GroupNorm(min(32, base_ch * 4), base_ch * 4),
            nn.SiLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1),
            nn.GroupNorm(min(32, base_ch * 4), base_ch * 4),
            nn.SiLU(),
        )

        # Time conditioning injection
        self.time_proj2 = nn.Linear(time_dim, base_ch * 2)
        self.time_proj3 = nn.Linear(time_dim, base_ch * 4)
        self.time_proj4 = nn.Linear(time_dim, base_ch * 4)

        # Global average pooling + classifier head
        feat_dim = base_ch * 4
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns log density ratio r(x_t, t) = log p_fake(x_t) / p_real(x_t)."""
        t_emb = self.time_embed(t)

        h = self.conv1(x_t)
        h = self.conv2(h)
        h = h + self.time_proj2(t_emb)[:, :, None, None]
        h = self.conv3(h)
        h = h + self.time_proj3(t_emb)[:, :, None, None]
        h = self.conv4(h)
        h = h + self.time_proj4(t_emb)[:, :, None, None]

        # Global average pool
        h = h.mean(dim=[2, 3])
        return self.head(h).squeeze(-1)


class SinPosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)
