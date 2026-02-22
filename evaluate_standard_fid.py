"""Standard FID evaluation: 50K generated vs 50K CIFAR-10 training images.

This follows the standard protocol used by all published CIFAR-10 results.
"""

import os
import sys
import json
import argparse
import gc

import torch
import torchvision
import torchvision.transforms as T
import numpy as np

from diffusers import DDPMPipeline, UNet2DModel

sys.path.insert(0, os.path.dirname(__file__))
from adadmd.evaluation.fid import InceptionV3Features, compute_stats, frechet_distance


class HFOneStepGenerator(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, z, y=None):
        t = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        out = self.unet(z, t, class_labels=y)
        return out.sample


@torch.no_grad()
def extract_features_from_generator(generator, inception, num_samples, device, batch_size=64):
    """Generate samples and extract Inception features."""
    generator.eval()
    all_feats = []
    n = 0
    while n < num_samples:
        bs = min(batch_size, num_samples - n)
        z = torch.randn(bs, 3, 32, 32, device=device)
        samples = generator(z)
        samples = ((samples + 1) / 2).clamp(0, 1)
        feats = inception(samples).cpu().numpy()
        all_feats.append(feats)
        n += bs
        if n % 5000 == 0:
            print(f"  Generated {n}/{num_samples}")
    return np.concatenate(all_feats)[:num_samples]


@torch.no_grad()
def extract_features_from_dataset(dataloader, inception, num_samples, device):
    """Extract Inception features from dataset."""
    all_feats = []
    n = 0
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        if imgs.min() < 0:
            imgs = (imgs + 1) / 2
        imgs = imgs.clamp(0, 1)
        feats = inception(imgs).cpu().numpy()
        all_feats.append(feats)
        n += len(imgs)
        if n >= num_samples:
            break
        if n % 10000 == 0:
            print(f"  Processed {n}/{num_samples} real images")
    return np.concatenate(all_feats)[:num_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_standard")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=50000)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Load generator
    pipe = DDPMPipeline.from_pretrained('google/ddpm-cifar10-32', torch_dtype=torch.float32)
    gen_unet = pipe.unet.to(device)
    del pipe
    gc.collect()

    if 'ema_generator' in ckpt:
        ema_state = ckpt['ema_generator']
        raw_state = ema_state.get('shadow', ema_state)
        clean_state = {k.replace('unet.', ''): v for k, v in raw_state.items()}
        gen_unet.load_state_dict(clean_state)
    elif 'generator' in ckpt:
        gen_state = ckpt['generator']
        clean_state = {k.replace('unet.', ''): v for k, v in gen_state.items()}
        gen_unet.load_state_dict(clean_state)

    generator = HFOneStepGenerator(gen_unet)
    generator.eval()

    # Load Inception
    print("Loading InceptionV3...")
    inception = InceptionV3Features(device=device)

    # Load CIFAR-10 TRAINING set (standard protocol)
    transform = T.Compose([T.ToTensor()])  # [0, 1]
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=False, num_workers=2
    )

    # Extract real features
    print(f"Extracting features from {args.num_samples} training images...")
    real_feats = extract_features_from_dataset(train_loader, inception, args.num_samples, device)

    # Generate and extract fake features
    print(f"Generating {args.num_samples} samples and extracting features...")
    fake_feats = extract_features_from_generator(
        generator, inception, args.num_samples, device
    )

    # Compute FID
    mu_real, sigma_real = compute_stats(real_feats)
    mu_fake, sigma_fake = compute_stats(fake_feats)
    fid = frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    results = {
        "fid_50k": round(fid, 2),
        "num_real": len(real_feats),
        "num_fake": len(fake_feats),
        "protocol": "50K generated vs 50K CIFAR-10 train (standard)"
    }

    with open(os.path.join(args.output_dir, 'results_standard.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nStandard FID (50K/50K): {fid:.2f}")
    print(f"Results saved to {args.output_dir}/results_standard.json")


if __name__ == "__main__":
    main()
