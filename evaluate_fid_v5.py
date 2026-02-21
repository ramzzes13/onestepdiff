"""Evaluate FID for AdaDMD v5 checkpoints.

Generates samples from checkpoints and computes FID against CIFAR-10 test set.
Also evaluates density-ratio verification.
"""

import os
import sys
import json
import argparse
import gc

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import numpy as np

from diffusers import UNet2DModel

sys.path.insert(0, os.path.dirname(__file__))
from adadmd.models.density_ratio import DensityRatioNetwork
from adadmd.evaluation.fid import compute_fid as compute_fid_from_tensors


class HFOneStepGenerator(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, z, y=None):
        t = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        out = self.unet(z, t, class_labels=y)
        return out.sample


@torch.no_grad()
def generate_samples(generator, num_samples, device, batch_size=64):
    """Generate samples from generator."""
    generator.eval()
    all_samples = []
    for i in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - i)
        z = torch.randn(bs, 3, 32, 32, device=device)
        samples = generator(z)
        samples = ((samples + 1) / 2).clamp(0, 1)
        all_samples.append(samples.cpu())
    return torch.cat(all_samples, dim=0)


@torch.no_grad()
def generate_with_dr_verification(generator, dr_net, num_samples, n_candidates,
                                   device, batch_size=64):
    """Generate samples with density-ratio verification (best-of-N)."""
    generator.eval()
    dr_net.eval()
    all_samples = []

    for i in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - i)
        best = None
        best_score = None

        for _ in range(n_candidates):
            z = torch.randn(bs, 3, 32, 32, device=device)
            samples = generator(z)
            t_zero = torch.zeros(bs, dtype=torch.long, device=device)
            scores = -dr_net(samples, t_zero)  # Higher = more real-like

            if best is None:
                best = samples
                best_score = scores
            else:
                mask = scores > best_score
                best[mask] = samples[mask]
                best_score[mask] = scores[mask]

        best = ((best + 1) / 2).clamp(0, 1)
        all_samples.append(best.cpu())

    return torch.cat(all_samples, dim=0)


def evaluate_checkpoint(ckpt_path, output_dir, device, num_samples=5000):
    """Evaluate a single checkpoint."""
    from diffusers import DDPMPipeline

    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Load generator
    pipe = DDPMPipeline.from_pretrained('google/ddpm-cifar10-32', torch_dtype=torch.float32)
    gen_unet = pipe.unet.to(device)
    del pipe
    gc.collect()

    # Load either EMA or regular generator weights
    if 'ema_generator' in ckpt:
        ema_state = ckpt['ema_generator']
        if 'shadow' in ema_state:
            gen_unet.load_state_dict(ema_state['shadow'])
        else:
            gen_unet.load_state_dict(ema_state)
    elif 'generator' in ckpt:
        gen_state = ckpt['generator']
        # Strip 'unet.' prefix if present
        clean_state = {}
        for k, v in gen_state.items():
            clean_state[k.replace('unet.', '')] = v
        gen_unet.load_state_dict(clean_state)

    generator = HFOneStepGenerator(gen_unet)
    generator.eval()

    # Load DR network
    dr_net = DensityRatioNetwork(in_channels=3, base_ch=32, image_size=32).to(device)
    if 'dr_net' in ckpt:
        dr_net.load_state_dict(ckpt['dr_net'])
    dr_net.eval()

    # Load CIFAR-10 test set
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    real_images = []
    for imgs, _ in test_loader:
        real_images.append((imgs + 1) / 2)  # [0, 1]
    real_images = torch.cat(real_images, dim=0)[:num_samples]

    results = {}
    os.makedirs(output_dir, exist_ok=True)

    # Standard FID
    print(f"Generating {num_samples} samples...")
    fake_images = generate_samples(generator, num_samples, device)
    fid = compute_fid_from_tensors(real_images, fake_images, device=device)
    results['fid'] = round(fid, 2)
    print(f"FID: {fid:.2f}")

    # Save sample grid
    vutils.save_image(fake_images[:64], os.path.join(output_dir, 'samples.png'), nrow=8)

    # DR-verified FID (N=4)
    print("Generating DR-verified samples (N=4)...")
    fake_dr4 = generate_with_dr_verification(generator, dr_net, num_samples, 4, device)
    fid_dr4 = compute_fid_from_tensors(real_images, fake_dr4, device=device)
    results['fid_dr4'] = round(fid_dr4, 2)
    print(f"FID (DR-verified N=4): {fid_dr4:.2f}")

    # DR-verified FID (N=8)
    print("Generating DR-verified samples (N=8)...")
    fake_dr8 = generate_with_dr_verification(generator, dr_net, num_samples, 8, device)
    fid_dr8 = compute_fid_from_tensors(real_images, fake_dr8, device=device)
    results['fid_dr8'] = round(fid_dr8, 2)
    print(f"FID (DR-verified N=8): {fid_dr8:.2f}")

    # Output stats
    with torch.no_grad():
        z = torch.randn(256, 3, 32, 32, device=device)
        x = generator(z)
        results['output_std'] = round(x.std().item(), 4)
        results['output_mean'] = round(x.mean().item(), 4)

    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults: {json.dumps(results, indent=2)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_v5")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=5000)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    evaluate_checkpoint(args.checkpoint, args.output_dir, device, args.num_samples)
