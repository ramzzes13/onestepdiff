"""Evaluate trained AdaDMD generator on CIFAR-10.

Computes FID, generates sample grids, and runs density-ratio verification.
"""

import os
import sys
import json
import torch
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
from adadmd.evaluation.dr_verification import batch_verified_generation


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/cifar10/checkpoints/final.pt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="outputs/cifar10/eval")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("AdaDMD CIFAR-10 Evaluation")
    print("=" * 60)

    # Load models
    print("Loading models...")
    base_model = SmallUNet(in_channels=3, out_channels=3, base_ch=64, num_classes=10).to(device)
    base_model.load_state_dict(
        torch.load('outputs/pretrain_cifar10/ddpm_final.pt', map_location=device, weights_only=True)
    )
    base_model.eval()

    generator = OneStepGenerator.from_pretrained_unet(base_model).to(device)
    dr_net = DensityRatioNetwork(in_channels=3, base_ch=32, image_size=32).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Try loading EMA generator first
    if 'ema_generator' in ckpt:
        ema_state = ckpt['ema_generator']
        generator.load_state_dict(ema_state)
        print("Loaded EMA generator weights")
    elif 'generator' in ckpt:
        generator.load_state_dict(ckpt['generator'])
        print("Loaded generator weights")

    dr_net.load_state_dict(ckpt['dr_net'])
    generator.eval()
    dr_net.eval()
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}, rank={ckpt.get('rank', '?')}")

    # Get real test images
    print(f"\nCollecting {args.num_samples} real test images...")
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)
    real_images = []
    for imgs, _ in test_loader:
        real_images.append(imgs)
        if sum(i.shape[0] for i in real_images) >= args.num_samples:
            break
    real_images = torch.cat(real_images)[:args.num_samples]

    # Generate fake images
    print(f"Generating {args.num_samples} fake images...")
    fake_images = []
    batch_size = 128
    for i in range(0, args.num_samples, batch_size):
        bs = min(batch_size, args.num_samples - i)
        z = torch.randn(bs, 3, 32, 32, device=device)
        labels = torch.randint(0, 10, (bs,), device=device)
        with torch.no_grad():
            samples = generator(z, labels)
        fake_images.append(samples.cpu())
    fake_images = torch.cat(fake_images)[:args.num_samples]

    # Save sample grid
    grid_samples = (fake_images[:64] + 1) / 2
    grid_samples = grid_samples.clamp(0, 1)
    vutils.save_image(grid_samples, os.path.join(args.output_dir, 'samples.png'), nrow=8)
    print(f"Saved sample grid to {args.output_dir}/samples.png")

    # Compute FID
    print("\nComputing FID...")
    fid = compute_fid(real_images, fake_images, device=str(device), batch_size=64)
    print(f"FID: {fid:.2f}")

    results = {
        'fid': fid,
        'num_samples': args.num_samples,
        'checkpoint': args.checkpoint,
        'step': ckpt.get('step', None),
        'lora_rank': ckpt.get('rank', None),
    }

    # Density-ratio verification
    print("\n--- Density-Ratio Verification ---")
    for n_candidates in [1, 4, 8]:
        verified_images = []
        for i in range(0, args.num_samples, batch_size):
            bs = min(batch_size, args.num_samples - i)
            labels = torch.randint(0, 10, (bs,), device=device)
            imgs, scores = batch_verified_generation(
                generator, dr_net, bs, (3, 32, 32),
                n_candidates=n_candidates, labels=labels, device=device
            )
            verified_images.append(imgs.cpu())
        verified_images = torch.cat(verified_images)[:args.num_samples]
        fid_v = compute_fid(real_images, verified_images, device=str(device), batch_size=64)
        print(f"  N_candidates={n_candidates}: FID = {fid_v:.2f}")
        results[f'fid_verified_n{n_candidates}'] = fid_v

    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"FID (no verification):  {results['fid']:.2f}")
    for n in [1, 4, 8]:
        k = f'fid_verified_n{n}'
        if k in results:
            print(f"FID (top-1 of {n}):      {results[k]:.2f}")
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    main()
