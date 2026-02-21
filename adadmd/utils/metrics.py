"""Evaluation metrics: FID computation using torchmetrics or clean-fid."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


def compute_fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute FID from precomputed statistics."""
    diff = mu1 - mu2
    covmean, _ = _sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        covmean = _sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])) @ sigma2)[0]

    fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(np.real(fid))


def _sqrtm(mat):
    """Matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, 0)
    sqrt_mat = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    return sqrt_mat, None


@torch.no_grad()
def compute_inception_features(images: torch.Tensor, inception_model: nn.Module,
                                batch_size: int = 64, device: torch.device = None) -> np.ndarray:
    """Compute inception features for a batch of images."""
    if device is None:
        device = next(inception_model.parameters()).device

    all_features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        # Resize to 299x299 for inception
        if batch.shape[-1] != 299 or batch.shape[-2] != 299:
            batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        # Normalize to [0, 1] if needed
        if batch.min() < 0:
            batch = (batch + 1) / 2
        features = inception_model(batch)
        if isinstance(features, tuple):
            features = features[0]
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def compute_statistics(features: np.ndarray):
    """Compute mean and covariance of features."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


@torch.no_grad()
def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor,
                inception_model: nn.Module, batch_size: int = 64,
                device: torch.device = None) -> float:
    """Compute FID between two sets of images."""
    real_features = compute_inception_features(real_images, inception_model, batch_size, device)
    fake_features = compute_inception_features(fake_images, inception_model, batch_size, device)

    mu_real, sigma_real = compute_statistics(real_features)
    mu_fake, sigma_fake = compute_statistics(fake_features)

    return compute_fid_from_stats(mu_real, sigma_real, mu_fake, sigma_fake)


@torch.no_grad()
def compute_inception_score(images: torch.Tensor, inception_model: nn.Module,
                             batch_size: int = 64, splits: int = 10,
                             device: torch.device = None) -> tuple:
    """Compute Inception Score."""
    if device is None:
        device = next(inception_model.parameters()).device

    all_probs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        if batch.shape[-1] != 299 or batch.shape[-2] != 299:
            batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        if batch.min() < 0:
            batch = (batch + 1) / 2
        logits = inception_model(batch)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    scores = []
    split_size = len(all_probs) // splits
    for i in range(splits):
        part = all_probs[i * split_size: (i + 1) * split_size]
        kl = part * (np.log(part + 1e-10) - np.log(np.mean(part, axis=0, keepdims=True) + 1e-10))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    return float(np.mean(scores)), float(np.std(scores))
