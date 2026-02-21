"""CIFAR-10 evaluation: FID, Inception Score, and density-ratio verification."""

import os
import sys
import torch
import numpy as np
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def get_test_images(num_samples=10000):
    """Get CIFAR-10 test images normalized to [-1, 1]."""
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=500, shuffle=False)
    images, labels = [], []
    for imgs, lbls in loader:
        images.append(imgs)
        labels.append(lbls)
        if sum(i.shape[0] for i in images) >= num_samples:
            break
    return torch.cat(images)[:num_samples], torch.cat(labels)[:num_samples]


@torch.no_grad()
def generate_samples(generator, num_samples=10000, batch_size=128,
                     device='cuda', num_classes=10):
    """Generate samples from the one-step generator."""
    generator.eval()
    all_samples = []
    for i in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - i)
        z = torch.randn(bs, 3, 32, 32, device=device)
        labels = torch.randint(0, num_classes, (bs,), device=device)
        samples = generator(z, labels)
        all_samples.append(samples.cpu())
    generator.train()
    return torch.cat(all_samples)


def evaluate_full(generator, dr_net=None, device='cuda', num_samples=5000):
    """Run full evaluation suite."""
    from .fid import compute_fid

    print(f"Evaluating with {num_samples} samples...")

    # Get real images
    real_images, real_labels = get_test_images(num_samples)

    # Generate fake images
    print("Generating samples...")
    fake_images = generate_samples(generator, num_samples, device=device)

    # Compute FID
    print("Computing FID...")
    fid = compute_fid(real_images, fake_images, device=device, batch_size=64)
    print(f"FID: {fid:.2f}")

    results = {'fid': fid, 'num_samples': num_samples}

    # Density-ratio verification experiment
    if dr_net is not None:
        print("\nDensity-ratio verification experiment:")
        for n_candidates in [1, 4, 8]:
            from .dr_verification import batch_verified_generation
            verified_images = []
            batch_size = 128
            for i in range(0, num_samples, batch_size):
                bs = min(batch_size, num_samples - i)
                labels = torch.randint(0, 10, (bs,), device=device)
                imgs, scores = batch_verified_generation(
                    generator, dr_net, bs, (3, 32, 32),
                    n_candidates=n_candidates, labels=labels, device=device
                )
                verified_images.append(imgs.cpu())
            verified_images = torch.cat(verified_images)[:num_samples]

            fid_verified = compute_fid(real_images, verified_images, device=device, batch_size=64)
            print(f"  N={n_candidates}: FID = {fid_verified:.2f}")
            results[f'fid_verified_n{n_candidates}'] = fid_verified

    return results
