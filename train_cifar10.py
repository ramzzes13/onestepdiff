"""Main script: Train AdaDMD on CIFAR-10.

Steps:
1. Train (or load) base DDPM on CIFAR-10
2. Run AdaDMD distillation to create one-step generator
3. Evaluate FID
"""

import os
import sys
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adadmd.configs.cifar10 import CIFAR10Config
from adadmd.training.pretrain_ddpm import load_or_train_base_model
from adadmd.training.trainer import AdaDMDTrainer
from adadmd.evaluation.fid import compute_fid


def get_cifar10_dataloader(batch_size=64, train=True):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=train, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                     num_workers=4, pin_memory=True, drop_last=train)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AdaDMD CIFAR-10 Training")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--pretrain_epochs", type=int, default=50,
                        help="Epochs for base DDPM pretraining")
    parser.add_argument("--distill_iters", type=int, default=50000,
                        help="Iterations for AdaDMD distillation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    config = CIFAR10Config()
    config.gpu_id = args.gpu
    config.batch_size = args.batch_size
    config.num_iterations = args.distill_iters
    config.device = device

    print("=" * 60)
    print("AdaDMD: Adaptive Distribution Matching Distillation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Distillation iterations: {config.num_iterations}")

    # Step 1: Load or train base DDPM
    print("\n--- Step 1: Base DDPM ---")
    base_model = load_or_train_base_model(
        output_dir="outputs/pretrain_cifar10",
        device=device,
        num_epochs=args.pretrain_epochs,
        base_channels=config.base_channels
    )

    # Step 2: Initialize AdaDMD trainer
    print("\n--- Step 2: Initialize AdaDMD ---")
    trainer = AdaDMDTrainer(config)
    # Copy base model weights into trainer
    trainer.base_model.load_state_dict(base_model.state_dict())
    # Freeze base model
    for p in trainer.base_model.parameters():
        p.requires_grad_(False)
    # Re-init generator from base model
    import copy
    trainer.generator.unet = copy.deepcopy(trainer.base_model)
    trainer.ema_generator = __import__('adadmd.utils.ema', fromlist=['EMA']).EMA(
        trainer.generator, decay=config.ema_decay
    )

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Step 3: Distillation
    if not args.eval_only:
        print("\n--- Step 3: AdaDMD Distillation ---")
        dataloader = get_cifar10_dataloader(batch_size=config.batch_size, train=True)
        trainer.train(dataloader, num_iterations=config.num_iterations)
        trainer.save_checkpoint()

    # Step 4: Evaluation
    print("\n--- Step 4: Evaluation ---")
    eval_dataloader = get_cifar10_dataloader(batch_size=64, train=False)

    # Collect real images
    real_images = []
    for images, _ in eval_dataloader:
        real_images.append(images)
        if len(real_images) * images.shape[0] >= config.num_eval_samples:
            break
    real_images = torch.cat(real_images)[:config.num_eval_samples]

    # Generate fake images
    print(f"Generating {config.num_eval_samples} samples...")
    fake_images = trainer.generate_samples(config.num_eval_samples)

    # Compute FID
    print("Computing FID...")
    fid = compute_fid(real_images, fake_images, device=device, batch_size=64)
    print(f"\n{'='*60}")
    print(f"FID Score: {fid:.2f}")
    print(f"{'='*60}")

    # Save results
    results = {
        'fid': fid,
        'num_eval_samples': config.num_eval_samples,
        'num_iterations': config.num_iterations,
        'lora_rank_final': trainer.fake_score_model.rank,
    }
    torch.save(results, os.path.join(config.output_dir, 'results.pt'))
    print(f"Results saved to {config.output_dir}/results.pt")

    return fid


if __name__ == "__main__":
    main()
