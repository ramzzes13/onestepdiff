"""Pretrain a DDPM on CIFAR-10 (or use HuggingFace pretrained model).

This script trains the base diffusion model that AdaDMD will distill.
For rapid prototyping, we also support loading a pretrained model from
HuggingFace Diffusers.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from adadmd.models.unet import SmallUNet
from adadmd.utils.diffusion import DiffusionSchedule
from adadmd.utils.ema import EMA


def train_ddpm_cifar10(
    output_dir: str = "outputs/pretrain_cifar10",
    num_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 2e-4,
    ema_decay: float = 0.9999,
    device: str = "cuda:0",
    base_channels: int = 64,
    num_timesteps: int = 1000,
    save_interval: int = 10,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    device = torch.device(device)

    # Dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=True)

    # Model
    model = SmallUNet(in_channels=3, out_channels=3, base_ch=base_channels,
                      num_classes=10).to(device)
    ema = EMA(model, decay=ema_decay)

    # Diffusion
    diffusion = DiffusionSchedule(num_timesteps=num_timesteps, schedule_type='linear').to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {num_epochs} epochs on CIFAR-10")

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        start = time.time()

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Sample timesteps
            t = torch.randint(0, num_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            x_t = diffusion.q_sample(images, t, noise)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                # Predict noise (epsilon prediction)
                pred_noise = model(x_t, t, labels)
                loss = F.mse_loss(pred_noise, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

        scheduler.step()
        elapsed = time.time() - start
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
              f"Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % save_interval == 0:
            # Save checkpoint
            torch.save(model.state_dict(),
                       os.path.join(output_dir, f'ddpm_epoch_{epoch+1}.pt'))
            torch.save(ema.state_dict(),
                       os.path.join(output_dir, f'ddpm_ema_epoch_{epoch+1}.pt'))

            # Generate samples with EMA model
            generate_ddpm_samples(ema.shadow, diffusion, device,
                                 os.path.join(output_dir, 'samples', f'epoch_{epoch+1}.png'))

    # Save final
    torch.save(model.state_dict(), os.path.join(output_dir, 'ddpm_final.pt'))
    torch.save(ema.state_dict(), os.path.join(output_dir, 'ddpm_ema_final.pt'))
    print("Training complete!")
    return model, ema


@torch.no_grad()
def generate_ddpm_samples(model, diffusion, device, save_path, num_samples=64,
                          num_classes=10, num_timesteps=1000):
    """Generate samples using DDPM reverse process."""
    import torchvision.utils as vutils
    model.eval()

    x = torch.randn(num_samples, 3, 32, 32, device=device)
    labels = torch.arange(num_classes, device=device).repeat(num_samples // num_classes + 1)[:num_samples]

    for t_idx in reversed(range(num_timesteps)):
        t = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
        pred_noise = model(x, t, labels)

        alpha_t = diffusion.alphas[t_idx]
        alpha_bar_t = diffusion.alphas_cumprod[t_idx]
        beta_t = diffusion.betas[t_idx]

        # DDPM reverse step
        mean = (1 / alpha_t.sqrt()) * (x - (beta_t / (1 - alpha_bar_t).sqrt()) * pred_noise)

        if t_idx > 0:
            noise = torch.randn_like(x)
            sigma = beta_t.sqrt()
            x = mean + sigma * noise
        else:
            x = mean

    x = (x + 1) / 2  # [-1, 1] -> [0, 1]
    x = x.clamp(0, 1)
    vutils.save_image(x, save_path, nrow=8)
    print(f"Saved DDPM samples to {save_path}")


def load_or_train_base_model(output_dir="outputs/pretrain_cifar10", device="cuda:0",
                              num_epochs=50, base_channels=64):
    """Load existing base model or train from scratch."""
    ema_path = os.path.join(output_dir, 'ddpm_ema_final.pt')
    model_path = os.path.join(output_dir, 'ddpm_final.pt')

    model = SmallUNet(in_channels=3, out_channels=3, base_ch=base_channels,
                      num_classes=10)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded pretrained DDPM from {model_path}")
        return model.to(device)

    print("No pretrained model found. Training DDPM from scratch...")
    model, ema = train_ddpm_cifar10(
        output_dir=output_dir, num_epochs=num_epochs,
        device=device, base_channels=base_channels
    )
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="outputs/pretrain_cifar10")
    parser.add_argument("--base_channels", type=int, default=64)
    args = parser.parse_args()

    train_ddpm_cifar10(
        output_dir=args.output_dir, num_epochs=args.epochs,
        batch_size=args.batch_size, device=args.device,
        base_channels=args.base_channels
    )
