"""Adaptive Distribution Matching Loss.

The core DMD loss: gradient of KL divergence expressed as difference of scores.
With adaptive weighting from the density-ratio network.
"""

import torch
import torch.nn as nn


class AdaptiveDMLoss(nn.Module):
    """Distribution Matching loss with adaptive timestep weighting.

    L_DM = E_t[w_t * ||s_real(x_t, t) - s_fake(x_t, t)||^2]

    where s_real and s_fake are score estimates from the real (frozen) and
    fake (LoRA) diffusion models, and w_t is the adaptive weight from
    the density-ratio network.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x_fake: torch.Tensor, s_real: torch.Tensor,
                s_fake: torch.Tensor, sigma_t: torch.Tensor,
                alpha_t: torch.Tensor, density_ratio_mag: torch.Tensor = None,
                ema_density_ratio: float = 1.0) -> torch.Tensor:
        """Compute adaptive DM loss.

        The DMD gradient for the generator is:
            dL/dx_fake = w_t * (s_fake(x_t, t) - s_real(x_t, t))

        We implement this as a loss whose gradient gives this update.

        Args:
            x_fake: Generated images (B, C, H, W)
            s_real: Real score estimate (B, C, H, W) - from frozen base model
            s_fake: Fake score estimate (B, C, H, W) - from LoRA model
            sigma_t: Noise level sigma_t for each sample (B,)
            alpha_t: Signal level alpha_t for each sample (B,)
            density_ratio_mag: |r_psi(x_t, t)| for adaptive weighting (B,)
            ema_density_ratio: EMA of density ratio magnitude (scalar)
        """
        # Score difference: direction to update generator
        score_diff = s_fake - s_real  # (B, C, H, W)

        # Base weighting: sigma_t^2 / alpha_t (as in DMD)
        base_weight = (sigma_t ** 2 / (alpha_t + self.epsilon))  # (B,)

        if density_ratio_mag is not None:
            # Adaptive weighting: downweight well-aligned timesteps
            adaptive_weight = base_weight / (ema_density_ratio + self.epsilon)
        else:
            adaptive_weight = base_weight

        adaptive_weight = adaptive_weight.view(-1, 1, 1, 1)

        # The loss is formulated so that grad_x_fake(L) = w_t * (s_fake - s_real)
        # This is achieved by: L = sum(x_fake * stopgrad(w_t * score_diff))
        weighted_diff = adaptive_weight * score_diff.detach()
        loss = (x_fake * weighted_diff).sum() / x_fake.shape[0]

        return loss
