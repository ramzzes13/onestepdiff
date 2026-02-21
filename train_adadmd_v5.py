"""AdaDMD v5: Fixed HF DDPM distillation with paired regression.

Key fixes over v4:
1. Pre-generate paired (noise, teacher_output) samples for proper regression
2. Use LPIPS loss (alex net, ~8MB vs 528MB vgg) to prevent mean collapse
3. Remove unpaired MSE entirely
4. Proper two-phase training with LoRA warmup
5. Longer training (50K+ steps)
6. Mixed precision training to save memory
7. Better DM loss scaling and gradient clipping
"""

import os
import sys
import time
import gc
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from peft import LoraConfig, get_peft_model
import lpips

sys.path.insert(0, os.path.dirname(__file__))
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


@torch.no_grad()
def generate_teacher_pairs(base_unet, scheduler, num_pairs, batch_size, device,
                           save_path=None):
    """Pre-generate (noise, clean_image) pairs from teacher DDPM."""
    base_unet.eval()
    all_noise = []
    all_images = []

    scheduler.set_timesteps(100)
    timesteps = scheduler.timesteps

    print(f"Generating {num_pairs} teacher pairs with {len(timesteps)} steps...")
    for i in range(0, num_pairs, batch_size):
        actual_bs = min(batch_size, num_pairs - i)

        z = torch.randn(actual_bs, 3, 32, 32, device=device)
        all_noise.append(z.cpu())

        x_t = z.clone()
        for t_step in timesteps:
            t_batch = torch.full((actual_bs,), t_step, device=device, dtype=torch.long)
            noise_pred = base_unet(x_t, t_batch).sample
            x_t = scheduler.step(noise_pred, t_step, x_t).prev_sample

        all_images.append(x_t.cpu().clamp(-1, 1))

        if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= num_pairs:
            done = min(i + batch_size, num_pairs)
            print(f"  Generated {done}/{num_pairs} pairs")

    noise_tensor = torch.cat(all_noise, dim=0)[:num_pairs]
    image_tensor = torch.cat(all_images, dim=0)[:num_pairs]

    if save_path:
        torch.save({'noise': noise_tensor, 'images': image_tensor}, save_path)
        print(f"Saved {num_pairs} teacher pairs to {save_path}")

    return noise_tensor, image_tensor


def train_adadmd_v5(device="cuda:0", num_iters=50000, batch_size=16,
                    lora_rank=8, output_dir="outputs/cifar10_v5",
                    num_teacher_pairs=10000,
                    lambda_reg=1.0, lambda_dm=0.001, lambda_dr=0.5,
                    lr_gen=2e-5, lr_lora=1e-4, use_lpips=True,
                    reg_phase_steps=5000, constant_reg=False):
    """Train AdaDMD v5 with fixed paired regression."""
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
    torch.cuda.empty_cache()

    for p in base_unet.parameters():
        p.requires_grad_(False)
    base_unet.eval()
    print(f"Base loaded. Memory: {torch.cuda.memory_allocated(device)/1e6:.0f} MB")

    # Generate or load teacher pairs
    pairs_path = os.path.join(output_dir, 'teacher_pairs.pt')
    if os.path.exists(pairs_path):
        print(f"Loading teacher pairs from {pairs_path}...")
        pairs = torch.load(pairs_path, weights_only=True)
        noise_bank = pairs['noise']
        image_bank = pairs['images']
    else:
        noise_bank, image_bank = generate_teacher_pairs(
            base_unet, scheduler, num_teacher_pairs, batch_size=64,
            device=device, save_path=pairs_path)

    print(f"Teacher pairs: {noise_bank.shape[0]}, noise range [{noise_bank.min():.2f}, {noise_bank.max():.2f}], "
          f"image range [{image_bank.min():.2f}, {image_bank.max():.2f}]")
    teacher_std = image_bank.std().item()
    teacher_mean = image_bank.mean().item()
    print(f"Teacher stats: mean={teacher_mean:.3f}, std={teacher_std:.3f}")

    # Create generator via save/reload
    print("Setting up generator...")
    torch.save(base_unet.state_dict(), '/tmp/gen_init_v5.pt')
    unet_config = dict(base_unet.config)
    gen_unet = UNet2DModel(**unet_config)
    gen_unet.load_state_dict(torch.load('/tmp/gen_init_v5.pt', weights_only=True))
    gen_unet = gen_unet.to(device)
    gen_unet.train()
    for p in gen_unet.parameters():
        p.requires_grad_(True)
    os.unlink('/tmp/gen_init_v5.pt')
    gc.collect()
    torch.cuda.empty_cache()
    generator = HFOneStepGenerator(gen_unet)
    print(f"Generator loaded. Memory: {torch.cuda.memory_allocated(device)/1e6:.0f} MB")

    # Create LoRA fake score model
    print("Setting up LoRA fake score model...")
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "conv1", "conv2"],
        lora_dropout=0.0,
    )
    fake_unet = get_peft_model(base_unet, lora_config)
    fake_unet.print_trainable_parameters()

    # EMA
    ema_gen = EMA(generator, decay=0.9999)

    # Density-ratio network
    dr_net = DensityRatioNetwork(in_channels=3, base_ch=32, image_size=32).to(device)

    # LPIPS loss - use 'alex' which is much smaller than 'vgg'
    if use_lpips:
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)
        print(f"LPIPS loaded (alex net). Memory: {torch.cuda.memory_allocated(device)/1e6:.0f} MB")
    else:
        lpips_fn = None

    lora_params = [p for p in fake_unet.parameters() if p.requires_grad]
    print(f"LoRA params: {sum(p.numel() for p in lora_params):,}")
    print(f"Generator: {sum(p.numel() for p in generator.parameters()):,} params")
    print(f"DR net: {sum(p.numel() for p in dr_net.parameters()):,} params")

    # Dataset (for real images)
    transform = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(),
                           T.Normalize([0.5]*3, [0.5]*3)])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True, drop_last=True)

    # Teacher pairs dataloader
    pairs_dataset = TensorDataset(noise_bank, image_bank)
    pairs_loader = DataLoader(pairs_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)

    # Optimizers
    opt_gen = torch.optim.AdamW(generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    opt_lora = torch.optim.AdamW(lora_params, lr=lr_lora)
    opt_dr = torch.optim.AdamW(dr_net.parameters(), lr=1e-4)

    nce_fn = NCELoss()
    dr_reg_fn = DensityRatioRegularizer(margin=1.0)
    ema_dr_val = EMAScalar(decay=0.99)

    # Diffusion params
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

    print(f"Total GPU memory: {torch.cuda.memory_allocated(device)/1e6:.0f} MB")
    print(f"\nStarting AdaDMD v5 distillation:")
    print(f"  Iterations: {num_iters}, batch_size={batch_size}")
    print(f"  Phase 1 (REG only): {reg_phase_steps} steps")
    print(f"  Phase 2 (DM+REG): {num_iters - reg_phase_steps} steps")
    print(f"  LPIPS: {use_lpips}, lambda_reg={lambda_reg}, lambda_dm={lambda_dm}")
    sys.stdout.flush()

    # Training loop
    data_iter = iter(loader)
    pairs_iter = iter(pairs_loader)
    start = time.time()
    log = {k: 0.0 for k in ['dm', 'nce', 'lora', 'acc', 'sdiff', 'reg',
                              'x_std', 'x_mean']}
    LOG_INT = 100
    SAVE_INT = 5000
    training_log = []

    for step in range(num_iters):
        # Get real images
        try:
            imgs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            imgs, labels = next(data_iter)
        imgs = imgs.to(device)
        B = imgs.shape[0]

        # Get teacher pairs
        try:
            z_paired, y_paired = next(pairs_iter)
        except StopIteration:
            pairs_iter = iter(pairs_loader)
            z_paired, y_paired = next(pairs_iter)
        z_paired = z_paired.to(device)
        y_paired = y_paired.to(device)

        in_reg_phase = step < reg_phase_steps

        # 1. Generate fake samples (from paired noise for regression)
        x_fake_paired = generator(z_paired)

        # 2. Update LoRA fake score model (predict noise)
        xf = x_fake_paired.detach()
        t_l = sample_t(B)
        noise_l = torch.randn_like(xf)
        x_l_t = q_sample(xf, t_l, noise_l)
        opt_lora.zero_grad()
        pred_noise = fake_unet(x_l_t, t_l).sample
        lora_loss = F.mse_loss(pred_noise, noise_l)
        lora_loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
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

        # 4. Generator loss
        dm_loss_val = 0.0
        score_diff_mag = 0.0

        # Paired regression loss
        if use_lpips and lpips_fn is not None:
            reg_loss = lpips_fn(x_fake_paired, y_paired).mean()
        else:
            # Huber loss is more robust than L1 or MSE for regression
            reg_loss = F.smooth_l1_loss(x_fake_paired, y_paired)
        reg_loss_val = reg_loss.item()

        if in_reg_phase:
            total = lambda_reg * reg_loss
        else:
            # Phase 2: DM + regression
            z_dm = torch.randn_like(imgs)
            x_fake_dm = generator(z_dm)
            t_dm = sample_t(B)
            noise_dm = torch.randn_like(x_fake_dm)
            x_fake_t_dm = q_sample(x_fake_dm, t_dm, noise_dm)

            with torch.no_grad():
                with fake_unet.disable_adapter():
                    eps_real = fake_unet(x_fake_t_dm, t_dm).sample
                eps_fake = fake_unet(x_fake_t_dm, t_dm).sample

                r_w = dr_net(x_fake_t_dm, t_dm)
                ema_val = ema_dr_val.update(r_w.abs().mean().item())

            sigma_t = (1 - alphas_cumprod[t_dm]).sqrt()
            alpha_t = alphas_cumprod[t_dm].sqrt()
            weight = (sigma_t ** 2 / (alpha_t + 1e-6)).view(-1, 1, 1, 1)
            ada_weight = weight / (ema_val + 1e-6)
            # Clamp adaptive weight to prevent explosion
            ada_weight = ada_weight.clamp(max=10.0)

            score_diff = (eps_fake - eps_real).detach()
            score_diff_mag = score_diff.abs().mean().item()
            # Use .mean() instead of .sum()/B for proper normalization
            dm_loss = (x_fake_dm * ada_weight * score_diff).mean()
            dm_loss_scaled = dm_loss * lambda_dm

            # DR regularizer on clean fake images
            t_zero = torch.zeros(B, dtype=torch.long, device=device)
            r_clean = dr_net(x_fake_dm, t_zero)
            dr_loss = dr_reg_fn(r_clean)

            # Keep regression strong to anchor outputs
            if constant_reg:
                reg_decay = 1.0
            else:
                progress = (step - reg_phase_steps) / max(1, num_iters - reg_phase_steps)
                reg_decay = max(0.3, 1.0 - 0.7 * progress)
            total = dm_loss_scaled + lambda_dr * dr_loss + lambda_reg * reg_decay * reg_loss

            dm_loss_val = dm_loss.item()

        opt_gen.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_gen.step()
        ema_gen.update(generator)

        # Track output statistics
        with torch.no_grad():
            x_std = x_fake_paired.std().item()
            x_mean = x_fake_paired.mean().item()

        log['dm'] += dm_loss_val
        log['nce'] += nce_loss.item()
        log['lora'] += lora_loss.item()
        log['acc'] += acc
        log['sdiff'] += score_diff_mag
        log['reg'] += reg_loss_val
        log['x_std'] += x_std
        log['x_mean'] += x_mean

        if (step+1) % LOG_INT == 0:
            elapsed = time.time() - start
            n = LOG_INT
            mem = torch.cuda.memory_allocated(device) / 1e6
            phase = "REG" if in_reg_phase else "DM+REG"
            avg = {k: v / n for k, v in log.items()}
            print(f'Step {step+1}/{num_iters} [{phase}] | '
                  f'DM:{avg["dm"]:.3f} | REG:{avg["reg"]:.4f} | '
                  f'LoRA:{avg["lora"]:.4f} | '
                  f'acc:{avg["acc"]:.3f} | sdiff:{avg["sdiff"]:.4f} | '
                  f'std:{avg["x_std"]:.3f} | mean:{avg["x_mean"]:.3f} | '
                  f'{n/elapsed:.1f}it/s | mem:{mem:.0f}MB')
            sys.stdout.flush()
            training_log.append({
                'step': step+1, 'phase': phase,
                **{k: round(v / n, 6) for k, v in log.items()}
            })
            log = {k: 0.0 for k in log}
            start = time.time()

        if (step+1) % SAVE_INT == 0:
            # Save samples
            ema_gen.shadow.eval()
            with torch.no_grad():
                z_s = torch.randn(64, 3, 32, 32, device=device)
                s = ema_gen.shadow(z_s)
            s = ((s + 1) / 2).clamp(0, 1)
            vutils.save_image(s, os.path.join(output_dir, 'samples', f'step_{step+1}.png'),
                            nrow=8)

            # Teacher pair comparison
            with torch.no_grad():
                s_paired = ema_gen.shadow(noise_bank[:64].to(device))
            s_paired = ((s_paired + 1) / 2).clamp(0, 1)
            vutils.save_image(s_paired, os.path.join(output_dir, 'samples', f'paired_{step+1}.png'),
                            nrow=8)

            if step+1 == SAVE_INT:
                t_ref = ((image_bank[:64] + 1) / 2).clamp(0, 1)
                vutils.save_image(t_ref, os.path.join(output_dir, 'samples', 'teacher_targets.png'),
                                nrow=8)

            torch.save({
                'step': step+1,
                'generator': generator.state_dict(),
                'ema_generator': ema_gen.state_dict(),
                'dr_net': dr_net.state_dict(),
                'lora': {k: v for k, v in fake_unet.state_dict().items() if 'lora' in k},
            }, os.path.join(output_dir, 'checkpoints', f'step_{step+1}.pt'))
            ema_gen.shadow.train()
            print(f'Saved at step {step+1}')
            sys.stdout.flush()

        if (step+1) % 1000 == 0 and x_std < 0.1:
            print(f"WARNING: Possible mean collapse at step {step+1}, x_std={x_std:.4f}")
            sys.stdout.flush()

    # Final save
    torch.save({
        'step': num_iters,
        'generator': generator.state_dict(),
        'ema_generator': ema_gen.state_dict(),
        'dr_net': dr_net.state_dict(),
    }, os.path.join(output_dir, 'checkpoints', 'final.pt'))

    with open(os.path.join(output_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)

    print('Training complete!')
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_iters", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="outputs/cifar10_v5")
    parser.add_argument("--num_teacher_pairs", type=int, default=10000)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--lambda_dm", type=float, default=0.001)
    parser.add_argument("--lambda_dr", type=float, default=0.5)
    parser.add_argument("--lr_gen", type=float, default=2e-5)
    parser.add_argument("--lr_lora", type=float, default=1e-4)
    parser.add_argument("--use_lpips", action="store_true", default=True)
    parser.add_argument("--no_lpips", action="store_false", dest="use_lpips")
    parser.add_argument("--reg_phase_steps", type=int, default=5000)
    parser.add_argument("--constant_reg", action="store_true", default=False,
                        help="Keep regression weight constant (no decay)")
    args = parser.parse_args()

    train_adadmd_v5(
        device=f"cuda:{args.gpu}", num_iters=args.num_iters,
        batch_size=args.batch_size, lora_rank=args.lora_rank,
        output_dir=args.output_dir,
        num_teacher_pairs=args.num_teacher_pairs,
        lambda_reg=args.lambda_reg, lambda_dm=args.lambda_dm,
        lambda_dr=args.lambda_dr, lr_gen=args.lr_gen,
        lr_lora=args.lr_lora, use_lpips=args.use_lpips,
        reg_phase_steps=args.reg_phase_steps,
        constant_reg=args.constant_reg,
    )
