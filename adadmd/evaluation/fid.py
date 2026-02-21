"""FID evaluation using torchvision's InceptionV3 or clean-fid."""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models


class InceptionV3Features(nn.Module):
    """InceptionV3 feature extractor for FID computation."""

    def __init__(self, device='cpu'):
        super().__init__()
        inception = models.inception_v3(weights='DEFAULT', transform_input=False)
        # Remove final classification layer, keep up to avgpool
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        # x: (B, 3, H, W) in [0, 1]
        # Resize to 299x299
        if x.shape[-1] != 299:
            x = torch.nn.functional.interpolate(x, size=(299, 299),
                                                 mode='bilinear', align_corners=False)
        features = self.blocks(x)
        return features.flatten(1)  # (B, 2048)


def compute_stats(features: np.ndarray):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute FID from statistics."""
    diff = mu1 - mu2

    # Product might be almost singular
    covmean_sq = sigma1 @ sigma2
    # Use eigendecomposition for matrix sqrt
    eigvals, eigvecs = np.linalg.eigh(covmean_sq)
    eigvals = np.maximum(eigvals, 0)
    covmean = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean_sq = (sigma1 + offset) @ (sigma2 + offset)
        eigvals, eigvecs = np.linalg.eigh(covmean_sq)
        eigvals = np.maximum(eigvals, 0)
        covmean = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(np.real(fid))


@torch.no_grad()
def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor,
                device='cuda', batch_size=64) -> float:
    """Compute FID between two sets of images.

    Args:
        real_images: (N, 3, H, W) in [0, 1] or [-1, 1]
        fake_images: (N, 3, H, W) in [0, 1] or [-1, 1]
    """
    inception = InceptionV3Features(device=device)

    # Normalize to [0, 1]
    if real_images.min() < 0:
        real_images = (real_images + 1) / 2
    if fake_images.min() < 0:
        fake_images = (fake_images + 1) / 2

    real_images = real_images.clamp(0, 1)
    fake_images = fake_images.clamp(0, 1)

    # Extract features
    real_feats = []
    for i in range(0, len(real_images), batch_size):
        batch = real_images[i:i+batch_size].to(device)
        feats = inception(batch).cpu().numpy()
        real_feats.append(feats)
    real_feats = np.concatenate(real_feats)

    fake_feats = []
    for i in range(0, len(fake_images), batch_size):
        batch = fake_images[i:i+batch_size].to(device)
        feats = inception(batch).cpu().numpy()
        fake_feats.append(feats)
    fake_feats = np.concatenate(fake_feats)

    mu_real, sigma_real = compute_stats(real_feats)
    mu_fake, sigma_fake = compute_stats(fake_feats)

    fid = frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    del inception
    torch.cuda.empty_cache()

    return fid


@torch.no_grad()
def compute_fid_from_generator(generator, dataloader, device, num_samples=5000,
                                batch_size=64) -> float:
    """Compute FID by generating samples from the generator and comparing to dataset."""
    inception = InceptionV3Features(device=device)

    # Collect real features
    real_feats = []
    n_collected = 0
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        if images.min() >= 0:
            images = images  # already [0, 1]
        else:
            images = (images + 1) / 2
        images = images.clamp(0, 1).to(device)
        feats = inception(images).cpu().numpy()
        real_feats.append(feats)
        n_collected += len(images)
        if n_collected >= num_samples:
            break
    real_feats = np.concatenate(real_feats)[:num_samples]

    # Generate fake samples and extract features
    generator.eval()
    fake_feats = []
    n_generated = 0
    while n_generated < num_samples:
        bs = min(batch_size, num_samples - n_generated)
        z = torch.randn(bs, 3, generator.unet.inc.weight.shape[1],
                        32, device=device)  # Will be set properly
        # Get image size from config
        img_size = 32  # default
        z = torch.randn(bs, 3, img_size, img_size, device=device)
        samples = generator(z)
        samples = (samples + 1) / 2
        samples = samples.clamp(0, 1)
        feats = inception(samples).cpu().numpy()
        fake_feats.append(feats)
        n_generated += bs
    fake_feats = np.concatenate(fake_feats)[:num_samples]

    mu_real, sigma_real = compute_stats(real_feats)
    mu_fake, sigma_fake = compute_stats(fake_feats)

    fid = frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    del inception
    torch.cuda.empty_cache()
    generator.train()

    return fid
