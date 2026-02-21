"""Hybrid regularization loss combining LPIPS and density-ratio components."""

import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    """Hybrid regularization: reduced LPIPS + density-ratio regularizer.

    L_reg = lambda_lpips * L_LPIPS + lambda_dr * L_DR
    """

    def __init__(self, lambda_lpips: float = 0.1, lambda_dr: float = 0.5,
                 use_lpips: bool = True):
        super().__init__()
        self.lambda_lpips = lambda_lpips
        self.lambda_dr = lambda_dr
        self.use_lpips = use_lpips

        if use_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
                for p in self.lpips_fn.parameters():
                    p.requires_grad_(False)
            except ImportError:
                print("Warning: lpips not installed, using L1 perceptual proxy")
                self.lpips_fn = None
                self.use_lpips = False

    def forward(self, x_fake: torch.Tensor = None, x_target: torch.Tensor = None,
                dr_loss: torch.Tensor = None) -> torch.Tensor:
        """Compute hybrid regularization loss.

        Args:
            x_fake: Generated images for LPIPS (B, C, H, W)
            x_target: Target images for LPIPS (B, C, H, W)
            dr_loss: Pre-computed density ratio loss (scalar)
        """
        loss = torch.tensor(0.0, device=dr_loss.device if dr_loss is not None
                           else x_fake.device)

        if self.use_lpips and x_fake is not None and x_target is not None:
            if self.lpips_fn is not None:
                self.lpips_fn = self.lpips_fn.to(x_fake.device)
                lpips_val = self.lpips_fn(x_fake, x_target).mean()
            else:
                # Fallback: L1 loss
                lpips_val = torch.nn.functional.l1_loss(x_fake, x_target)
            loss = loss + self.lambda_lpips * lpips_val

        if dr_loss is not None:
            loss = loss + self.lambda_dr * dr_loss

        return loss
