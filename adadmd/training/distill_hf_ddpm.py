"""AdaDMD distillation using HuggingFace pretrained DDPM.

Uses google/ddpm-cifar10-32 as the base model, with PEFT LoRA for
the fake score model. The base model uses epsilon (noise) prediction.
"""

import os
import sys
import time
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from adadmd.models.density_ratio import DensityRatioNetwork
from adadmd.losses.nce_loss import NCELoss
from adadmd.losses.dr_loss import DensityRatioRegularizer
from adadmd.utils.ema import EMA, EMAScalar


class HFOneStepGenerator(nn.Module):
    """One-step generator wrapping HF UNet2DModel with fixed t=0."""

    def __init__(self, unet: UNet2DModel):
        super().__init__()
        self.unet = unet

    def forward(self, z, y=None):
        t = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        out = self.unet(z, t, class_labels=y)
        return out.sample


def train_adadmd_hf(device="cuda:0", num_iters=30000, batch_size=8,
                     lora_rank=8, output_dir="outputs/cifar10_hf_v2",
                     lora_warmup=2000):
    """Train AdaDMD using HF pretrained DDPM.

    Key fixes over v1:
    - LoRA targets conv layers (conv1, conv2) in addition to attention
    - LoRA loss predicts noise (epsilon), matching model parameterization
    - LoRA warmup phase: first lora_warmup steps only train LoRA, freeze gen
    - Memory-efficient loading (save/reload instead of deepcopy)
    """
    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    # Load pretrained DDPM
    print("Loading google/ddpm-cifar10-32...")
    pipe = DDPMPipeline.from_pretrained('google/ddpm-cifar10-32', torch_dtype=torch.float32)
    base_unet = pipe.unet.to(device)
    scheduler = pipe.scheduler
    del pipe
    gc.collect()

    # Freeze base model and save weights for generator init
    for p in base_unet.parameters():
        p.requires_grad_(False)
    base_unet.eval()
    print(f"Base loaded. Memory: {torch.cuda.memory_allocated(device)/1e6:.0f} MB")

    # Create generator via save/reload (avoids deepcopy memory spike)
    print("Setting up generator...")
    torch.save(base_unet.state_dict(), '/tmp/gen_init.pt')
    unet_config = dict(base_unet.config)
    gen_unet = UNet2DModel(**unet_config)
    gen_unet.load_state_dict(torch.load('/tmp/gen_init.pt', weights_only=True))
    gen_unet = gen_unet.to(device)
    gen_unet.train()
    for p in gen_unet.parameters():
        p.requires_grad_(True)
    os.unlink('/tmp/gen_init.pt')
    gc.collect()
    torch.cuda.empty_cache()
    generator = HFOneStepGenerator(gen_unet)
    print(f"Generator loaded. Memory: {torch.cuda.memory_allocated(device)/1e6:.0f} MB")

    # Create LoRA fake score model - target conv AND attention layers
    # PEFT modifies base_unet in-place, so we use disable_adapter() for base output
    print("Setting up LoRA fake score model...")
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "conv1", "conv2"],
        lora_dropout=0.0,
    )
    fake_unet = get_peft_model(base_unet, lora_config)
    fake_unet.print_trainable_parameters()

    # EMA
    ema_gen = EMA(generator, decay=0.9999)

    # Density-ratio network
    dr_net = DensityRatioNetwork(in_channels=3, base_ch=32, image_size=32).to(device)

    lora_params = [p for p in fake_unet.parameters() if p.requires_grad]
    print(f"LoRA params: {sum(p.numel() for p in lora_params):,}")
    print(f"Generator: {sum(p.numel() for p in generator.parameters()):,} params")
    print(f"DR net: {sum(p.numel() for p in dr_net.parameters()):,} params")
    print(f"Total GPU memory: {torch.cuda.memory_allocated(device)/1e6:.0f} MB")

    # Dataset
    transform = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(),
                           T.Normalize([0.5]*3, [0.5]*3)])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True, drop_last=True)

    # Optimizers
    opt_gen = torch.optim.AdamW(generator.parameters(), lr=2e-5, betas=(0.5, 0.999))
    opt_lora = torch.optim.AdamW(lora_params, lr=1e-4)
    opt_dr = torch.optim.AdamW(dr_net.parameters(), lr=1e-4)

    nce_fn = NCELoss()
    dr_reg_fn = DensityRatioRegularizer(margin=1.0)
    ema_dr = EMAScalar(decay=0.99)

    # Diffusion params from scheduler
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    num_timesteps = scheduler.config.num_train_timesteps

    def q_sample(x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def sample_t(B, t_min=0.02, t_max=0.98):
        low = int(t_min * num_timesteps)
        high = int(t_max * num_timesteps)
        return torch.randint(low, high, (B,), device=device)

    print(f"After optimizers. Memory: {torch.cuda.memory_allocated(device)/1e6:.0f} MB")
    print(f"\nStarting distillation ({num_iters} iters, bs={batch_size})...")
    print(f"LoRA warmup: {lora_warmup} steps (generator frozen)")

    # Training loop
    data_iter = iter(loader)
    start = time.time()
    log = {k: 0 for k in ['dm', 'nce', 'dr', 'lora', 'acc', 'sdiff']}
    LOG_INT = 100
    SAVE_INT = 5000

    for step in range(num_iters):
        try:
            imgs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            imgs, labels = next(data_iter)
        imgs = imgs.to(device)
        B = imgs.shape[0]
        in_warmup = step < lora_warmup

        # 1. Generate
        z = torch.randn_like(imgs)
        if in_warmup:
            with torch.no_grad():
                x_fake = generator(z)
        else:
            x_fake = generator(z)

        # 2. Update LoRA fake score (predict noise, NOT clean image)
        xf = x_fake.detach()
        t_l = sample_t(B)
        noise_l = torch.randn_like(xf)
        x_l_t = q_sample(xf, t_l, noise_l)
        opt_lora.zero_grad()
        pred_noise = fake_unet(x_l_t, t_l).sample
        lora_loss = F.mse_loss(pred_noise, noise_l)
        lora_loss.backward()
        opt_lora.step()

        # 3. Update DR network
        t = sample_t(B)
        noise = torch.randn_like(imgs)
        x_real_t = q_sample(imgs, t, noise)
        x_fake_t = q_sample(xf, t, noise)

        opt_dr.zero_grad()
        r_real = dr_net(x_real_t, t)
        r_fake = dr_net(x_fake_t, t)
        nce_loss = nce_fn(r_real, r_fake)
        nce_loss.backward()
        opt_dr.step()
        acc = nce_fn.accuracy(r_real.detach(), r_fake.detach())

        # 4. DM loss (skip during warmup)
        dm_loss_val = 0.0
        score_diff_mag = 0.0
        if not in_warmup:
            x_fake = generator(z)
            t_dm = sample_t(B)
            noise_dm = torch.randn_like(x_fake)
            x_fake_t_dm = q_sample(x_fake, t_dm, noise_dm)

            with torch.no_grad():
                # Both predict noise (epsilon parameterization)
                # Use disable_adapter to get base model output (no LoRA)
                with fake_unet.disable_adapter():
                    eps_real = fake_unet(x_fake_t_dm, t_dm).sample
                # With LoRA enabled for fake score
                eps_fake = fake_unet(x_fake_t_dm, t_dm).sample

                r_w = dr_net(x_fake_t_dm, t_dm)
                ema_val = ema_dr.update(r_w.abs().mean().item())

            sigma_t = (1 - alphas_cumprod[t_dm]).sqrt()
            alpha_t = alphas_cumprod[t_dm].sqrt()
            weight = (sigma_t ** 2 / (alpha_t + 1e-6)).view(-1, 1, 1, 1)
            ada_weight = weight / (ema_val + 1e-6)

            score_diff = (eps_fake - eps_real).detach()
            score_diff_mag = score_diff.abs().mean().item()
            dm_loss = (x_fake * ada_weight * score_diff).sum() / B

            # DR reg
            t_zero = torch.zeros(B, dtype=torch.long, device=device)
            r_clean = dr_net(x_fake, t_zero)
            dr_loss = dr_reg_fn(r_clean)
            total = dm_loss + 0.5 * dr_loss

            opt_gen.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_gen.step()
            ema_gen.update(generator)

            dm_loss_val = dm_loss.item()
        else:
            dr_loss = torch.tensor(0.0)

        log['dm'] += dm_loss_val
        log['nce'] += nce_loss.item()
        log['dr'] += dr_loss.item()
        log['lora'] += lora_loss.item()
        log['acc'] += acc
        log['sdiff'] += score_diff_mag

        if (step+1) % LOG_INT == 0:
            elapsed = time.time() - start
            n = LOG_INT
            mem = torch.cuda.memory_allocated(device) / 1e6
            phase = "WARMUP" if in_warmup else "TRAIN"
            print(f'Step {step+1}/{num_iters} [{phase}] | '
                  f'DM:{log["dm"]/n:.3f} | NCE:{log["nce"]/n:.4f} | '
                  f'DR:{log["dr"]/n:.3f} | LoRA:{log["lora"]/n:.4f} | '
                  f'acc:{log["acc"]/n:.3f} | sdiff:{log["sdiff"]/n:.6f} | '
                  f'{n/elapsed:.1f}it/s | mem:{mem:.0f}MB')
            sys.stdout.flush()
            log = {k: 0 for k in log}
            start = time.time()

        if (step+1) % SAVE_INT == 0:
            import torchvision.utils as vutils
            ema_gen.shadow.eval()
            with torch.no_grad():
                z_s = torch.randn(64, 3, 32, 32, device=device)
                s = ema_gen.shadow(z_s)
            s = ((s + 1) / 2).clamp(0, 1)
            vutils.save_image(s, os.path.join(output_dir, 'samples', f'step_{step+1}.png'), nrow=8)
            torch.save({
                'step': step+1,
                'generator': generator.state_dict(),
                'ema_generator': ema_gen.state_dict(),
                'dr_net': dr_net.state_dict(),
            }, os.path.join(output_dir, 'checkpoints', f'step_{step+1}.pt'))
            ema_gen.shadow.train()
            print(f'Saved at step {step+1}')
            sys.stdout.flush()

    # Final save
    torch.save({
        'step': num_iters,
        'generator': generator.state_dict(),
        'ema_generator': ema_gen.state_dict(),
        'dr_net': dr_net.state_dict(),
    }, os.path.join(output_dir, 'checkpoints', 'final.pt'))
    print('Training complete!')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_iters", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_warmup", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="outputs/cifar10_hf_v2")
    args = parser.parse_args()

    train_adadmd_hf(device=f"cuda:{args.gpu}", num_iters=args.num_iters,
                     batch_size=args.batch_size, lora_rank=args.lora_rank,
                     output_dir=args.output_dir, lora_warmup=args.lora_warmup)
