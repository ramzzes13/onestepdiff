"""AdaDMD Trainer: the main training loop.

Implements the full AdaDMD training algorithm:
1. Generate fake samples with generator
2. Update density-ratio network via NCE
3. Compute adaptive distribution matching loss
4. Compute hybrid regularization
5. Update generator
6. Update LoRA fake score model
7. Periodically increase LoRA rank
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from ..models.unet import SmallUNet
from ..models.lora import LoRAWrapper
from ..models.density_ratio import DensityRatioNetwork
from ..models.generator import OneStepGenerator
from ..losses.dm_loss import AdaptiveDMLoss
from ..losses.nce_loss import NCELoss
from ..losses.dr_loss import DensityRatioRegularizer
from ..losses.hybrid_loss import HybridLoss
from ..utils.diffusion import DiffusionSchedule
from ..utils.ema import EMA, EMAScalar


class AdaDMDTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.setup_models()
        self.setup_losses()
        self.setup_optimizers()
        self.setup_diffusion()

        self.global_step = 0
        self.ema_dr_scalar = EMAScalar(decay=config.ema_dr_decay)

        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'samples'), exist_ok=True)

    def setup_models(self):
        cfg = self.config
        # 1. Base diffusion model (frozen, serves as real score model)
        self.base_model = SmallUNet(
            in_channels=cfg.in_channels, out_channels=cfg.in_channels,
            base_ch=cfg.base_channels, num_classes=cfg.num_classes
        ).to(self.device)

        # 2. LoRA-wrapped fake score model
        self.fake_score_model = LoRAWrapper(
            self.base_model, rank=cfg.lora_rank_init,
            alpha=cfg.lora_alpha, target_modules=list(cfg.lora_target_modules)
        ).to(self.device)

        # 3. One-step generator (initialized from base model)
        self.generator = OneStepGenerator.from_pretrained_unet(self.base_model).to(self.device)

        # 4. Density-ratio network
        self.dr_network = DensityRatioNetwork(
            in_channels=cfg.in_channels, base_ch=cfg.dr_base_channels,
            image_size=cfg.image_size
        ).to(self.device)

        # EMA for generator
        self.ema_generator = EMA(self.generator, decay=cfg.ema_decay)

        print(f"Base model params: {sum(p.numel() for p in self.base_model.parameters()):,}")
        print(f"LoRA params: {self.fake_score_model.num_lora_params():,}")
        print(f"Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"DR network params: {sum(p.numel() for p in self.dr_network.parameters()):,}")

    def setup_losses(self):
        self.dm_loss_fn = AdaptiveDMLoss()
        self.nce_loss_fn = NCELoss(label_smoothing=0.0)
        self.dr_reg_fn = DensityRatioRegularizer(margin=self.config.dr_margin)
        self.hybrid_loss_fn = HybridLoss(
            lambda_lpips=self.config.lambda_lpips,
            lambda_dr=self.config.lambda_dr,
            use_lpips=self.config.use_lpips
        )

    def setup_optimizers(self):
        cfg = self.config
        # Generator optimizer
        self.opt_gen = torch.optim.AdamW(
            self.generator.parameters(), lr=cfg.lr_generator,
            weight_decay=cfg.weight_decay, betas=(0.5, 0.999)
        )
        # LoRA optimizer (fake score model)
        self.opt_lora = torch.optim.AdamW(
            self.fake_score_model.lora_parameters(), lr=cfg.lr_lora,
            weight_decay=cfg.weight_decay, betas=(0.9, 0.999)
        )
        # Density-ratio optimizer
        self.opt_dr = torch.optim.AdamW(
            self.dr_network.parameters(), lr=cfg.lr_density_ratio,
            weight_decay=cfg.weight_decay, betas=(0.9, 0.999)
        )

        if cfg.mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def setup_diffusion(self):
        cfg = self.config
        self.diffusion = DiffusionSchedule(
            num_timesteps=cfg.num_timesteps,
            schedule_type=cfg.schedule_type,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end
        ).to(self.device)

    def load_base_model(self, checkpoint_path: str):
        """Load pretrained base model weights."""
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.base_model.load_state_dict(state_dict)
        # Re-init generator from updated base model
        import copy
        self.generator.unet = copy.deepcopy(self.base_model)
        self.ema_generator = EMA(self.generator, decay=self.config.ema_decay)
        print(f"Loaded base model from {checkpoint_path}")

    def train_step(self, real_images: torch.Tensor, labels: torch.Tensor = None):
        """Single AdaDMD training step."""
        cfg = self.config
        B = real_images.shape[0]

        # ========== Step 1: Generate fake samples ==========
        z = torch.randn_like(real_images)
        with torch.no_grad() if not self.generator.training else torch.enable_grad():
            pass

        x_fake = self.generator(z, labels)

        # ========== Step 2: Update density-ratio network ==========
        t = self.diffusion.sample_timesteps(B, cfg.t_min, cfg.t_max, self.device)
        noise = torch.randn_like(real_images)

        x_real_t = self.diffusion.q_sample(real_images, t, noise)
        x_fake_t = self.diffusion.q_sample(x_fake.detach(), t, noise)

        self.opt_dr.zero_grad()
        if self.scaler:
            with autocast(device_type='cuda'):
                r_real = self.dr_network(x_real_t, t)
                r_fake = self.dr_network(x_fake_t, t)
                nce_loss = self.nce_loss_fn(r_real, r_fake)
            self.scaler.scale(nce_loss).backward()
            self.scaler.step(self.opt_dr)
        else:
            r_real = self.dr_network(x_real_t, t)
            r_fake = self.dr_network(x_fake_t, t)
            nce_loss = self.nce_loss_fn(r_real, r_fake)
            nce_loss.backward()
            self.opt_dr.step()

        dr_accuracy = self.nce_loss_fn.accuracy(r_real.detach(), r_fake.detach())

        # ========== Step 3: Compute adaptive distribution matching loss ==========
        # Re-generate fake with gradients
        x_fake = self.generator(z, labels)

        # Sample new timesteps for DM loss
        t_dm = self.diffusion.sample_timesteps(B, cfg.t_min, cfg.t_max, self.device)
        noise_dm = torch.randn_like(x_fake)
        x_fake_t_dm = self.diffusion.q_sample(x_fake, t_dm, noise_dm)

        # Real score (frozen base model)
        with torch.no_grad():
            s_real = self.base_model(x_fake_t_dm, t_dm, labels)

        # Fake score (LoRA model) - no gradient through this for generator
        with torch.no_grad():
            s_fake = self.fake_score_model(x_fake_t_dm, t_dm, labels)

        # Adaptive weighting from density ratio
        with torch.no_grad():
            r_for_weight = self.dr_network(x_fake_t_dm, t_dm)
            dr_mag = r_for_weight.abs()
            ema_dr = self.ema_dr_scalar.update(dr_mag.mean().item())

        sigma_t = self.diffusion.sigma_t[t_dm]
        alpha_t = self.diffusion.alpha_t[t_dm]

        dm_loss = self.dm_loss_fn(x_fake, s_real, s_fake, sigma_t, alpha_t, dr_mag, ema_dr)

        # ========== Step 4: Hybrid regularization ==========
        # DR regularizer on clean fake images (t=0)
        with torch.no_grad():
            t_zero = torch.zeros(B, dtype=torch.long, device=self.device)
        r_fake_clean = self.dr_network(x_fake, t_zero)
        dr_reg_loss = self.dr_reg_fn(r_fake_clean)
        reg_loss = self.hybrid_loss_fn(dr_loss=dr_reg_loss)

        # ========== Step 5: Update generator ==========
        total_gen_loss = dm_loss + reg_loss
        self.opt_gen.zero_grad()
        if self.scaler:
            self.scaler.scale(total_gen_loss).backward()
            self.scaler.step(self.opt_gen)
        else:
            total_gen_loss.backward()
            self.opt_gen.step()

        # Update EMA
        self.ema_generator.update(self.generator)

        # ========== Step 6: Update LoRA fake score model ==========
        x_fake_detached = x_fake.detach()
        t_lora = self.diffusion.sample_timesteps(B, cfg.t_min, cfg.t_max, self.device)
        noise_lora = torch.randn_like(x_fake_detached)
        x_lora_t = self.diffusion.q_sample(x_fake_detached, t_lora, noise_lora)

        self.opt_lora.zero_grad()
        if self.scaler:
            with autocast(device_type='cuda'):
                pred = self.fake_score_model(x_lora_t, t_lora, labels)
                # Train to predict noise (epsilon parameterization), NOT clean image
                lora_loss = F.mse_loss(pred, noise_lora)
            self.scaler.scale(lora_loss).backward()
            self.scaler.step(self.opt_lora)
        else:
            pred = self.fake_score_model(x_lora_t, t_lora, labels)
            # Train to predict noise (epsilon parameterization), NOT clean image
            lora_loss = F.mse_loss(pred, noise_lora)
            lora_loss.backward()
            self.opt_lora.step()

        if self.scaler:
            self.scaler.update()

        # ========== Step 7: Rank warming ==========
        self.global_step += 1
        if (cfg.rank_warm_interval > 0 and
            self.global_step % cfg.rank_warm_interval == 0 and
            self.fake_score_model.rank < cfg.lora_rank_max):
            new_rank = min(self.fake_score_model.rank * 2, cfg.lora_rank_max)
            self.fake_score_model.increase_rank(new_rank)
            # Re-create optimizer for new parameters
            self.opt_lora = torch.optim.AdamW(
                self.fake_score_model.lora_parameters(), lr=cfg.lr_lora,
                weight_decay=cfg.weight_decay, betas=(0.9, 0.999)
            )
            print(f"Step {self.global_step}: LoRA rank increased to {new_rank}")

        return {
            'dm_loss': dm_loss.item(),
            'nce_loss': nce_loss.item(),
            'dr_reg_loss': dr_reg_loss.item(),
            'lora_loss': lora_loss.item(),
            'total_gen_loss': total_gen_loss.item(),
            'dr_accuracy': dr_accuracy,
            'ema_dr': ema_dr,
            'lora_rank': self.fake_score_model.rank,
        }

    @torch.no_grad()
    def generate_samples(self, num_samples: int, labels: torch.Tensor = None,
                         use_ema: bool = True) -> torch.Tensor:
        """Generate samples using the generator."""
        model = self.ema_generator.shadow if use_ema else self.generator
        model.eval()

        all_samples = []
        batch_size = min(64, num_samples)
        for i in range(0, num_samples, batch_size):
            bs = min(batch_size, num_samples - i)
            z = torch.randn(bs, self.config.in_channels,
                           self.config.image_size, self.config.image_size,
                           device=self.device)
            if labels is not None:
                y = labels[i:i+bs]
            else:
                y = None
            samples = model(z, y)
            all_samples.append(samples.cpu())

        model.train()
        return torch.cat(all_samples, dim=0)

    def save_checkpoint(self, path: str = None):
        if path is None:
            path = os.path.join(self.config.output_dir, 'checkpoints',
                               f'step_{self.global_step}.pt')
        torch.save({
            'global_step': self.global_step,
            'generator': self.generator.state_dict(),
            'ema_generator': self.ema_generator.state_dict(),
            'lora': self.fake_score_model.lora_state_dict(),
            'dr_network': self.dr_network.state_dict(),
            'opt_gen': self.opt_gen.state_dict(),
            'opt_lora': self.opt_lora.state_dict(),
            'opt_dr': self.opt_dr.state_dict(),
            'lora_rank': self.fake_score_model.rank,
            'config': self.config,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.global_step = ckpt['global_step']
        self.generator.load_state_dict(ckpt['generator'])
        self.ema_generator.load_state_dict(ckpt['ema_generator'])
        self.fake_score_model.load_lora_state_dict(ckpt['lora'])
        self.dr_network.load_state_dict(ckpt['dr_network'])
        self.opt_gen.load_state_dict(ckpt['opt_gen'])
        self.opt_lora.load_state_dict(ckpt['opt_lora'])
        self.opt_dr.load_state_dict(ckpt['opt_dr'])
        print(f"Loaded checkpoint from {path}, step={self.global_step}")

    def train(self, dataloader: DataLoader, num_iterations: int = None):
        """Main training loop."""
        if num_iterations is None:
            num_iterations = self.config.num_iterations

        self.generator.train()
        self.dr_network.train()

        data_iter = iter(dataloader)
        start_time = time.time()
        log_losses = {}

        for step in range(self.global_step, num_iterations):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1] if len(batch) > 1 else None
            else:
                images, labels = batch, None

            images = images.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)

            # Normalize to [-1, 1] if needed
            if images.min() >= 0:
                images = images * 2 - 1

            metrics = self.train_step(images, labels)

            # Accumulate for logging
            for k, v in metrics.items():
                if k not in log_losses:
                    log_losses[k] = 0
                log_losses[k] += v

            if (step + 1) % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                it_per_sec = self.config.log_interval / elapsed
                avg = {k: v / self.config.log_interval for k, v in log_losses.items()}
                print(f"Step {step+1}/{num_iterations} | "
                      f"DM: {avg['dm_loss']:.4f} | NCE: {avg['nce_loss']:.4f} | "
                      f"DR_reg: {avg['dr_reg_loss']:.4f} | LoRA: {avg['lora_loss']:.4f} | "
                      f"DR_acc: {avg['dr_accuracy']:.3f} | "
                      f"rank: {int(avg['lora_rank'])} | "
                      f"{it_per_sec:.1f} it/s")
                log_losses = {}
                start_time = time.time()

            if (step + 1) % self.config.save_interval == 0:
                self.save_checkpoint()

            if (step + 1) % self.config.eval_interval == 0:
                self._eval_and_save_samples(step + 1)

    @torch.no_grad()
    def _eval_and_save_samples(self, step: int):
        """Generate and save sample images."""
        import torchvision.utils as vutils
        samples = self.generate_samples(64)
        samples = (samples + 1) / 2  # [-1,1] -> [0,1]
        samples = samples.clamp(0, 1)
        path = os.path.join(self.config.output_dir, 'samples', f'step_{step}.png')
        vutils.save_image(samples, path, nrow=8)
        print(f"Saved samples to {path}")
