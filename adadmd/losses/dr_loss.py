"""Density-Ratio Regularizer.

Penalizes generator for producing samples classified as fake by
the density-ratio network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityRatioRegularizer(nn.Module):
    """Density-ratio verification loss.

    L_DR = E_z[max(0, -r_psi(G(z), 0) + delta)]

    Penalizes generator when density ratio network confidently classifies
    outputs as fake (large negative r_psi).
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, density_ratio: torch.Tensor) -> torch.Tensor:
        """Compute DR regularizer.

        Args:
            density_ratio: r_psi(x_fake, t) values (B,)
                Positive = looks real, negative = looks fake

        Returns:
            DR regularization loss
        """
        # Hinge loss: penalize when r < -margin (strongly fake)
        loss = F.relu(-density_ratio + self.margin)
        return loss.mean()
