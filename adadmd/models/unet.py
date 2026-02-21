"""UNet models: wraps HuggingFace diffusers UNet2DModel with LoRA support."""

import torch
import torch.nn as nn
import math


class SinusoidalPosEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(h))))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        q = self.q(norm_x).reshape(b, self.num_heads, self.head_dim, h * w)
        k = self.k(norm_x).reshape(b, self.num_heads, self.head_dim, h * w)
        v = self.v(norm_x).reshape(b, self.num_heads, self.head_dim, h * w)
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.head_dim ** -0.5
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v).reshape(b, c, h, w)
        return x + self.out(out)


class SmallUNet(nn.Module):
    """Simple UNet for 32x32 images (CIFAR-10 scale).

    ~7M params with base_ch=64, suitable for training on limited GPU memory.
    """

    def __init__(self, in_channels=3, out_channels=3, base_ch=64, num_classes=0):
        super().__init__()
        time_dim = base_ch * 4

        self.time_embed = nn.Sequential(
            SinusoidalPosEmbed(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.num_classes = num_classes
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, time_dim)

        # Encoder: 32->16->8->4
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 2, base_ch * 4

        self.inc = nn.Conv2d(in_channels, c1, 3, padding=1)

        self.down1_r1 = ResBlock(c1, c1, time_dim)
        self.down1_r2 = ResBlock(c1, c1, time_dim)
        self.pool1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)

        self.down2_r1 = ResBlock(c1, c2, time_dim)
        self.down2_r2 = ResBlock(c2, c2, time_dim)
        self.pool2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)

        self.down3_r1 = ResBlock(c2, c3, time_dim)
        self.down3_attn = AttentionBlock(c3)
        self.down3_r2 = ResBlock(c3, c3, time_dim)
        self.pool3 = nn.Conv2d(c3, c3, 3, stride=2, padding=1)

        # Bottleneck at 4x4
        self.mid_r1 = ResBlock(c3, c4, time_dim)
        self.mid_attn = AttentionBlock(c4)
        self.mid_r2 = ResBlock(c4, c4, time_dim)

        # Decoder: 4->8->16->32
        self.up3_r1 = ResBlock(c4 + c3, c3, time_dim)
        self.up3_attn = AttentionBlock(c3)
        self.up3_r2 = ResBlock(c3, c3, time_dim)

        self.up2_r1 = ResBlock(c3 + c2, c2, time_dim)
        self.up2_r2 = ResBlock(c2, c2, time_dim)

        self.up1_r1 = ResBlock(c2 + c1, c1, time_dim)
        self.up1_r2 = ResBlock(c1, c1, time_dim)

        self.out_norm = nn.GroupNorm(min(32, c1), c1)
        self.out_conv = nn.Conv2d(c1, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, y=None):
        t_emb = self.time_embed(t)
        if y is not None and self.num_classes > 0:
            t_emb = t_emb + self.class_embed(y)

        # Encoder
        h1 = self.inc(x)
        h1 = self.down1_r2(self.down1_r1(h1, t_emb), t_emb)
        h2 = self.pool1(h1)

        h2 = self.down2_r2(self.down2_r1(h2, t_emb), t_emb)
        h3 = self.pool2(h2)

        h3 = self.down3_r1(h3, t_emb)
        h3 = self.down3_attn(h3)
        h3 = self.down3_r2(h3, t_emb)
        h4 = self.pool3(h3)

        # Bottleneck
        h4 = self.mid_r1(h4, t_emb)
        h4 = self.mid_attn(h4)
        h4 = self.mid_r2(h4, t_emb)

        # Decoder
        h = torch.nn.functional.interpolate(h4, scale_factor=2, mode='nearest')
        h = torch.cat([h, h3], dim=1)
        h = self.up3_r1(h, t_emb)
        h = self.up3_attn(h)
        h = self.up3_r2(h, t_emb)

        h = torch.nn.functional.interpolate(h, scale_factor=2, mode='nearest')
        h = torch.cat([h, h2], dim=1)
        h = self.up2_r1(h, t_emb)
        h = self.up2_r2(h, t_emb)

        h = torch.nn.functional.interpolate(h, scale_factor=2, mode='nearest')
        h = torch.cat([h, h1], dim=1)
        h = self.up1_r1(h, t_emb)
        h = self.up1_r2(h, t_emb)

        return self.out_conv(torch.nn.functional.silu(self.out_norm(h)))
