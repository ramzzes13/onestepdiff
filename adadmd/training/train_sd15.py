"""AdaDMD distillation for Stable Diffusion v1.5.

Uses HuggingFace diffusers for the base SD model with PEFT LoRA
for the fake score model. Operates in latent space.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from adadmd.utils.ema import EMA, EMAScalar


class LatentDensityRatioNet(nn.Module):
    """Density-ratio network operating in SD latent space (4-channel, 64x64)."""

    def __init__(self, in_channels=4, base_ch=32):
        super().__init__()
        import math

        class SinEmb(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
            def forward(self, t):
                half = self.dim // 2
                emb = math.log(10000) / (half - 1)
                emb = torch.exp(torch.arange(half, device=t.device) * -emb)
                emb = t[:, None].float() * emb[None, :]
                return torch.cat([emb.sin(), emb.cos()], dim=-1)

        time_dim = base_ch * 4
        self.time_embed = nn.Sequential(
            SinEmb(base_ch), nn.Linear(base_ch, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.GroupNorm(16, base_ch * 2), nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.GroupNorm(32, base_ch * 4), nn.SiLU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1),
            nn.GroupNorm(32, base_ch * 4), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.time_proj = nn.Linear(time_dim, base_ch * 4)
        self.head = nn.Sequential(
            nn.Linear(base_ch * 4, base_ch * 2), nn.SiLU(),
            nn.Linear(base_ch * 2, 1),
        )

    def forward(self, x_t, t):
        t_emb = self.time_embed(t)
        h = self.net(x_t).squeeze(-1).squeeze(-1)
        h = h + self.time_proj(t_emb)
        return self.head(h).squeeze(-1)


class PromptDataset(Dataset):
    """Simple dataset of text prompts."""

    def __init__(self, prompts_file: str = None, prompts: list = None):
        if prompts is not None:
            self.prompts = prompts
        elif prompts_file and os.path.exists(prompts_file):
            with open(prompts_file) as f:
                self.prompts = [line.strip() for line in f if line.strip()]
        else:
            # Default: use simple prompts for training
            self.prompts = [
                "a photo of a cat", "a photo of a dog", "a beautiful landscape",
                "a portrait of a person", "a city skyline at night",
                "a bowl of fruit on a table", "a red car on a highway",
                "a snowy mountain peak", "a sunset over the ocean",
                "a cozy living room with a fireplace",
            ] * 100

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


class SDAdaDMDTrainer:
    """AdaDMD trainer for Stable Diffusion v1.5 in latent space."""

    def __init__(self, device="cuda:0", lora_rank=8, lora_alpha=16,
                 lr_gen=1e-5, lr_lora=5e-5, lr_dr=1e-4,
                 lambda_dr=0.5, num_inference_steps=1000,
                 t_min=0.02, t_max=0.50, output_dir="outputs/sd15"):
        self.device = torch.device(device)
        self.lora_rank = lora_rank
        self.lr_gen = lr_gen
        self.lr_lora = lr_lora
        self.lr_dr = lr_dr
        self.lambda_dr = lambda_dr
        self.t_min = t_min
        self.t_max = t_max
        self.num_inference_steps = num_inference_steps
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

        self._load_models(lora_rank, lora_alpha)
        self._setup_optimizers()
        self.global_step = 0
        self.ema_dr_scalar = EMAScalar(decay=0.99)

    def _load_models(self, lora_rank, lora_alpha):
        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae",
                                                   torch_dtype=torch.float16)
        self.vae.to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        print("Loading text encoder...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float16
        )
        self.text_encoder.to(self.device).eval()
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        print("Loading UNet (real score model - frozen)...")
        self.real_unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float16
        )
        self.real_unet.to(self.device).eval()
        for p in self.real_unet.parameters():
            p.requires_grad_(False)

        print("Loading UNet (fake score model with LoRA)...")
        self.fake_unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float16
        )
        self.fake_unet.to(self.device)
        # Apply LoRA
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
        )
        self.fake_unet = get_peft_model(self.fake_unet, lora_config)
        self.fake_unet.print_trainable_parameters()
        # Freeze non-LoRA params
        for name, param in self.fake_unet.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad_(False)

        print("Loading UNet (generator)...")
        self.gen_unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float32
        )
        self.gen_unet.to(self.device).train()
        # Generator needs gradients
        for p in self.gen_unet.parameters():
            p.requires_grad_(True)

        # Noise scheduler
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # Density-ratio network
        self.dr_net = LatentDensityRatioNet(in_channels=4, base_ch=32).to(self.device)

        # Print memory
        mem = torch.cuda.memory_allocated(self.device) / 1e9
        print(f"GPU memory after loading: {mem:.2f} GB")

    def _setup_optimizers(self):
        self.opt_gen = torch.optim.AdamW(
            self.gen_unet.parameters(), lr=self.lr_gen, betas=(0.5, 0.999)
        )
        lora_params = [p for p in self.fake_unet.parameters() if p.requires_grad]
        self.opt_lora = torch.optim.AdamW(lora_params, lr=self.lr_lora)
        self.opt_dr = torch.optim.AdamW(self.dr_net.parameters(), lr=self.lr_dr)
        self.scaler = GradScaler()

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(prompt, padding="max_length", max_length=77,
                                truncation=True, return_tensors="pt")
        text_emb = self.text_encoder(tokens.input_ids.to(self.device))[0]
        return text_emb.half()

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space."""
        return self.vae.encode(image.half()).latent_dist.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        return self.vae.decode(latent.half() / self.vae.config.scaling_factor).sample

    def sample_timesteps(self, batch_size):
        T = self.scheduler.config.num_train_timesteps
        low = int(self.t_min * T)
        high = int(self.t_max * T)
        return torch.randint(low, high, (batch_size,), device=self.device)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion in latent space."""
        if noise is None:
            noise = torch.randn_like(x_0)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        sqrt_alpha = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def train_step(self, prompts: list):
        B = len(prompts)
        text_embs = torch.cat([self.encode_prompt(p) for p in prompts])

        # 1. Generate fake latents
        z = torch.randn(B, 4, 64, 64, device=self.device)
        # Generator operates at t=0 (single step: noise -> clean latent)
        t_zero = torch.zeros(B, dtype=torch.long, device=self.device)
        x_fake = self.gen_unet(z, t_zero, encoder_hidden_states=text_embs.float()).sample

        # 2. Update density-ratio network
        t = self.sample_timesteps(B)
        noise = torch.randn_like(x_fake)
        # Need real latents - use noise as proxy for real data distribution
        # (In full pipeline, we'd use actual image latents)
        x_real_proxy = torch.randn_like(x_fake) * 0.5  # Simplified real proxy
        x_real_t = self.q_sample(x_real_proxy, t, noise)
        x_fake_t = self.q_sample(x_fake.detach(), t, noise)

        self.opt_dr.zero_grad()
        r_real = self.dr_net(x_real_t, t)
        r_fake = self.dr_net(x_fake_t, t)
        nce_loss = -torch.log(torch.sigmoid(r_real) + 1e-8).mean() \
                   - torch.log(1 - torch.sigmoid(r_fake) + 1e-8).mean()
        nce_loss.backward()
        self.opt_dr.step()

        # 3. DM loss
        x_fake = self.gen_unet(z, t_zero, encoder_hidden_states=text_embs.float()).sample
        t_dm = self.sample_timesteps(B)
        noise_dm = torch.randn_like(x_fake)
        x_fake_t_dm = self.q_sample(x_fake, t_dm, noise_dm)

        with torch.no_grad():
            s_real = self.real_unet(x_fake_t_dm.half(), t_dm,
                                     encoder_hidden_states=text_embs).sample.float()
            s_fake = self.fake_unet(x_fake_t_dm.half(), t_dm,
                                     encoder_hidden_states=text_embs).sample.float()

        score_diff = (s_fake - s_real).detach()
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        sigma_t = (1 - alphas_cumprod[t_dm]).sqrt()
        alpha_t = alphas_cumprod[t_dm].sqrt()
        weight = (sigma_t ** 2 / (alpha_t + 1e-6)).view(-1, 1, 1, 1)

        dm_loss = (x_fake * weight * score_diff).sum() / B

        # 4. DR regularizer
        r_clean = self.dr_net(x_fake, t_zero)
        dr_loss = F.relu(-r_clean + 1.0).mean()
        total_loss = dm_loss + self.lambda_dr * dr_loss

        self.opt_gen.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gen_unet.parameters(), 1.0)
        self.opt_gen.step()

        # 5. Update LoRA fake score
        x_fake_det = x_fake.detach()
        t_lora = self.sample_timesteps(B)
        noise_lora = torch.randn_like(x_fake_det)
        x_lora_t = self.q_sample(x_fake_det, t_lora, noise_lora)

        self.opt_lora.zero_grad()
        pred = self.fake_unet(x_lora_t.half(), t_lora,
                               encoder_hidden_states=text_embs).sample.float()
        lora_loss = F.mse_loss(pred, x_fake_det)
        lora_loss.backward()
        self.opt_lora.step()

        self.global_step += 1
        return {
            'dm_loss': dm_loss.item(),
            'nce_loss': nce_loss.item(),
            'dr_loss': dr_loss.item(),
            'lora_loss': lora_loss.item(),
        }

    @torch.no_grad()
    def generate(self, prompt: str, num_images: int = 1):
        text_emb = self.encode_prompt(prompt)
        text_emb = text_emb.expand(num_images, -1, -1)
        z = torch.randn(num_images, 4, 64, 64, device=self.device)
        t_zero = torch.zeros(num_images, dtype=torch.long, device=self.device)
        latent = self.gen_unet(z, t_zero, encoder_hidden_states=text_emb.float()).sample
        images = self.decode_latent(latent)
        return (images.float().clamp(-1, 1) + 1) / 2

    def train(self, dataloader, num_steps=1000, log_interval=10):
        data_iter = iter(dataloader)
        start_time = time.time()

        for step in range(self.global_step, num_steps):
            try:
                prompts = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                prompts = next(data_iter)

            if isinstance(prompts, torch.Tensor):
                prompts = [f"image {i}" for i in range(len(prompts))]

            metrics = self.train_step(prompts)

            if (step + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Step {step+1}/{num_steps} | "
                      f"DM: {metrics['dm_loss']:.4f} | NCE: {metrics['nce_loss']:.4f} | "
                      f"DR: {metrics['dr_loss']:.4f} | LoRA: {metrics['lora_loss']:.4f} | "
                      f"{log_interval/elapsed:.1f} it/s")
                start_time = time.time()

            if (step + 1) % 100 == 0:
                self._save_samples(step + 1)

            if (step + 1) % 500 == 0:
                self._save_checkpoint(step + 1)

    def _save_samples(self, step):
        import torchvision.utils as vutils
        prompts = ["a photo of a cat", "a beautiful sunset", "a city skyline",
                    "a red sports car"]
        images = []
        for p in prompts:
            img = self.generate(p, num_images=1)
            images.append(img)
        images = torch.cat(images)
        path = os.path.join(self.output_dir, 'samples', f'step_{step}.png')
        vutils.save_image(images, path, nrow=2)
        print(f"Saved samples to {path}")

    def _save_checkpoint(self, step):
        path = os.path.join(self.output_dir, 'checkpoints', f'step_{step}.pt')
        torch.save({
            'step': step,
            'gen_unet': self.gen_unet.state_dict(),
            'fake_unet_lora': {k: v for k, v in self.fake_unet.state_dict().items()
                               if 'lora' in k.lower()},
            'dr_net': self.dr_net.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="outputs/sd15")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    trainer = SDAdaDMDTrainer(device=device, lora_rank=args.lora_rank,
                               output_dir=args.output_dir)

    dataset = PromptDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    trainer.train(dataloader, num_steps=args.num_steps)


if __name__ == "__main__":
    main()
