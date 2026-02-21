"""One-Step Generator for AdaDMD.

The generator maps noise z ~ N(0, I) to images in a single forward pass.
It reuses the UNet architecture but without time conditioning (or with fixed t=0).
"""

import torch
import torch.nn as nn
from .unet import SmallUNet


class OneStepGenerator(nn.Module):
    """One-step generator that maps noise to images.

    Architecture: same as SmallUNet but with time embedding fixed to t=0.
    Initialized from pretrained diffusion model weights.
    """

    def __init__(self, unet: SmallUNet):
        super().__init__()
        self.unet = unet
        # Register a fixed timestep of 0
        self.register_buffer('fixed_t', torch.zeros(1, dtype=torch.long))

    @classmethod
    def from_pretrained_unet(cls, unet: SmallUNet):
        """Create generator by copying weights from pretrained UNet."""
        import copy
        gen_unet = copy.deepcopy(unet)
        return cls(gen_unet)

    def forward(self, z: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """Generate images from noise.

        Args:
            z: Input noise, shape (B, C, H, W)
            y: Optional class labels, shape (B,)

        Returns:
            Generated images, shape (B, C, H, W)
        """
        t = self.fixed_t.expand(z.shape[0])
        return self.unet(z, t, y)
