"""Evaluate all trained AdaDMD models on CIFAR-10.

Computes FID for:
1. Custom UNet v4 distillation
2. HF pretrained DDPM v4 distillation
3. Density-ratio verification experiments
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adadmd.models.unet import SmallUNet
from adadmd.models.generator import OneStepGenerator
from adadmd.models.density_ratio import DensityRatioNetwork
from adadmd.evaluation.fid import compute_fid


class HFOneStepGenerator(nn.Module):
    """One-step generator wrapping HF UNet2DModel with fixed t=0."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, z, y=None):
        t = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        out = self.unet(z, t, class_labels=y)
        return out.sample


def get_real_images(num_samples=10000):
    """Get CIFAR-10 test images normalized to [-1, 1]."""
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=500, shuffle=False)
    images = []
    for imgs, _ in loader:
        images.append(imgs)
        if sum(i.shape[0] for i in images) >= num_samples:
            break
    return torch.cat(images)[:num_samples]


@torch.no_grad()
def generate_samples(generator, num_samples, batch_size=128, device='cuda'):
    """Generate samples from any one-step generator."""
    generator.eval()
    all_samples = []
    for i in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - i)
        z = torch.randn(bs, 3, 32, 32, device=device)
        samples = generator(z)
        all_samples.append(samples.cpu())
    return torch.cat(all_samples)[:num_samples]


def evaluate_custom_v4(device, real_images, num_samples):
    """Evaluate custom UNet v4 distillation."""
    results = {}

    # Find best checkpoint
    ckpt_dir = 'outputs/cifar10_v4/checkpoints'
    if not os.path.exists(ckpt_dir):
        print("  No custom v4 checkpoints found, skipping")
        return results

    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
    if not ckpts:
        print("  No checkpoints found")
        return results

    for ckpt_name in ckpts:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        print(f"\n  Evaluating {ckpt_name}...")

        # Load base model
        base_model = SmallUNet(in_channels=3, out_channels=3, base_ch=64, num_classes=10).to(device)
        base_state = torch.load('outputs/pretrain_cifar10_v2/ddpm_final.pt',
                                map_location=device, weights_only=True)
        base_model.load_state_dict(base_state)

        # Load generator
        generator = OneStepGenerator.from_pretrained_unet(base_model).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        if 'ema_generator' in ckpt:
            generator.load_state_dict(ckpt['ema_generator'])
            print("    Using EMA weights")
        elif 'generator' in ckpt:
            generator.load_state_dict(ckpt['generator'])
        generator.eval()

        # Generate and compute FID
        fake_images = generate_samples(generator, num_samples, batch_size=128, device=str(device))
        fid = compute_fid(real_images, fake_images, device=str(device), batch_size=32)
        step = ckpt.get('step', ckpt_name)
        print(f"    FID: {fid:.2f} (step {step})")
        results[f'custom_v4_{ckpt_name}'] = {'fid': fid, 'step': step}

        # Save sample grid
        grid = ((fake_images[:64] + 1) / 2).clamp(0, 1)
        os.makedirs('outputs/cifar10_v4/eval', exist_ok=True)
        vutils.save_image(grid, f'outputs/cifar10_v4/eval/samples_{ckpt_name}.png', nrow=8)

        # DR verification if available
        if 'dr_net' in ckpt:
            dr_net = DensityRatioNetwork(in_channels=3, base_ch=32, image_size=32).to(device)
            dr_net.load_state_dict(ckpt['dr_net'])
            dr_net.eval()

            from adadmd.evaluation.dr_verification import batch_verified_generation
            for n_cand in [4, 8]:
                verified = []
                for i in range(0, num_samples, 64):
                    bs = min(64, num_samples - i)
                    imgs, _ = batch_verified_generation(
                        generator, dr_net, bs, (3, 32, 32),
                        n_candidates=n_cand, device=str(device)
                    )
                    verified.append(imgs.cpu())
                verified = torch.cat(verified)[:num_samples]
                fid_v = compute_fid(real_images, verified, device=str(device), batch_size=32)
                print(f"    FID (top-1 of {n_cand}): {fid_v:.2f}")
                results[f'custom_v4_{ckpt_name}_verified_n{n_cand}'] = fid_v

            del dr_net

        del generator, base_model
        torch.cuda.empty_cache()

    return results


def evaluate_hf_v4(device, real_images, num_samples):
    """Evaluate HF pretrained DDPM v4 distillation."""
    results = {}

    ckpt_dir = 'outputs/cifar10_hf_v4/checkpoints'
    if not os.path.exists(ckpt_dir):
        print("  No HF v4 checkpoints found, skipping")
        return results

    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
    if not ckpts:
        print("  No checkpoints found")
        return results

    from diffusers import UNet2DModel

    for ckpt_name in ckpts:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        print(f"\n  Evaluating {ckpt_name}...")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Load HF UNet architecture
        from diffusers import DDPMPipeline
        pipe = DDPMPipeline.from_pretrained('google/ddpm-cifar10-32', torch_dtype=torch.float32)
        gen_unet = pipe.unet.to(device)
        del pipe

        generator = HFOneStepGenerator(gen_unet)

        # Load weights - handle EMA state
        if 'ema_generator' in ckpt:
            ema_state = ckpt['ema_generator']
            # EMA state has 'shadow' key containing the actual model state
            if isinstance(ema_state, dict) and 'shadow' in ema_state:
                shadow = ema_state['shadow']
                # Map shadow keys to generator keys
                gen_state = {}
                for k, v in shadow.items():
                    gen_state[k] = v
                generator.load_state_dict(gen_state)
            else:
                generator.load_state_dict(ema_state)
            print("    Using EMA weights")
        elif 'generator' in ckpt:
            generator.load_state_dict(ckpt['generator'])
        generator.eval()

        # Generate and compute FID
        fake_images = generate_samples(generator, num_samples, batch_size=128, device=str(device))
        fid = compute_fid(real_images, fake_images, device=str(device), batch_size=32)
        step = ckpt.get('step', ckpt_name)
        print(f"    FID: {fid:.2f} (step {step})")
        results[f'hf_v4_{ckpt_name}'] = {'fid': fid, 'step': step}

        # Save sample grid
        grid = ((fake_images[:64] + 1) / 2).clamp(0, 1)
        os.makedirs('outputs/cifar10_hf_v4/eval', exist_ok=True)
        vutils.save_image(grid, f'outputs/cifar10_hf_v4/eval/samples_{ckpt_name}.png', nrow=8)

        # DR verification
        if 'dr_net' in ckpt:
            dr_net = DensityRatioNetwork(in_channels=3, base_ch=32, image_size=32).to(device)
            dr_net.load_state_dict(ckpt['dr_net'])
            dr_net.eval()

            from adadmd.evaluation.dr_verification import batch_verified_generation
            for n_cand in [4, 8]:
                verified = []
                for i in range(0, num_samples, 64):
                    bs = min(64, num_samples - i)
                    imgs, _ = batch_verified_generation(
                        generator, dr_net, bs, (3, 32, 32),
                        n_candidates=n_cand, device=str(device)
                    )
                    verified.append(imgs.cpu())
                verified = torch.cat(verified)[:num_samples]
                fid_v = compute_fid(real_images, verified, device=str(device), batch_size=32)
                print(f"    FID (top-1 of {n_cand}): {fid_v:.2f}")
                results[f'hf_v4_{ckpt_name}_verified_n{n_cand}'] = fid_v

            del dr_net

        del generator
        torch.cuda.empty_cache()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    num_samples = args.num_samples

    print("=" * 70)
    print("AdaDMD CIFAR-10 Comprehensive Evaluation")
    print("=" * 70)

    # Get real images
    print(f"\nLoading {num_samples} real CIFAR-10 test images...")
    real_images = get_real_images(num_samples)
    print(f"Real images loaded: {real_images.shape}")

    all_results = {}

    # 1. Custom v4
    print("\n" + "=" * 70)
    print("1. Custom UNet v4 Distillation")
    print("=" * 70)
    custom_results = evaluate_custom_v4(device, real_images, num_samples)
    all_results.update(custom_results)

    # 2. HF v4
    print("\n" + "=" * 70)
    print("2. HuggingFace Pretrained DDPM v4 Distillation")
    print("=" * 70)
    hf_results = evaluate_hf_v4(device, real_images, num_samples)
    all_results.update(hf_results)

    # Save all results
    os.makedirs('outputs/eval_results', exist_ok=True)
    with open('outputs/eval_results/all_fid_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL RESULTS")
    print("=" * 70)
    for key, val in all_results.items():
        if isinstance(val, dict):
            print(f"  {key}: FID = {val.get('fid', 'N/A'):.2f}")
        else:
            print(f"  {key}: FID = {val:.2f}")

    print(f"\nResults saved to outputs/eval_results/all_fid_results.json")


if __name__ == "__main__":
    main()
