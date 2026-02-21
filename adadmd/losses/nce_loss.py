"""Noise Contrastive Estimation loss for density-ratio network training.

Trains the density-ratio network r_psi to distinguish real from fake noised samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCELoss(nn.Module):
    """Binary classification (NCE) loss for density-ratio estimation.

    L_NCE = -E_{x_t ~ p_real}[log sigma(r_psi(x_t, t))]
            -E_{x_t ~ p_fake}[log(1 - sigma(r_psi(x_t, t)))]

    where sigma is the sigmoid function.
    Convention: r_psi > 0 means "looks real", r_psi < 0 means "looks fake".
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, r_real: torch.Tensor, r_fake: torch.Tensor) -> torch.Tensor:
        """Compute NCE loss.

        Args:
            r_real: Density ratio output for real noised samples (B,)
            r_fake: Density ratio output for fake noised samples (B,)

        Returns:
            NCE loss scalar
        """
        # Real samples should have high r (classified as real)
        real_target = 1.0 - self.label_smoothing
        fake_target = self.label_smoothing

        loss_real = F.binary_cross_entropy_with_logits(
            r_real, torch.full_like(r_real, real_target)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            r_fake, torch.full_like(r_fake, fake_target)
        )

        return (loss_real + loss_fake) / 2

    @staticmethod
    def accuracy(r_real: torch.Tensor, r_fake: torch.Tensor) -> float:
        """Compute classification accuracy for monitoring."""
        with torch.no_grad():
            correct_real = (r_real > 0).float().mean()
            correct_fake = (r_fake < 0).float().mean()
            return ((correct_real + correct_fake) / 2).item()
