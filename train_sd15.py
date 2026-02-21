"""Main script: Train AdaDMD on Stable Diffusion v1.5.

This distills SD v1.5 into a one-step generator using AdaDMD.
Requires ~8GB GPU memory with fp16 and gradient checkpointing.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adadmd.training.train_sd15 import SDAdaDMDTrainer, PromptDataset


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AdaDMD SD v1.5 Distillation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="outputs/sd15")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    print("=" * 60)
    print("AdaDMD: SD v1.5 Distillation")
    print("=" * 60)

    trainer = SDAdaDMDTrainer(
        device=device, lora_rank=args.lora_rank,
        output_dir=args.output_dir
    )

    if not args.eval_only:
        dataset = PromptDataset()
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        trainer.train(dataloader, num_steps=args.num_steps)

    # Generate final samples
    print("\nGenerating evaluation samples...")
    import torchvision.utils as vutils
    prompts = [
        "a photo of a cat sitting on a windowsill",
        "a beautiful mountain landscape at sunset",
        "a city street in the rain at night",
        "a bowl of fresh fruit on a wooden table",
        "a portrait of a dog wearing a hat",
        "a cozy cabin in a snowy forest",
        "a sports car on a winding road",
        "flowers in a garden with butterflies",
    ]
    all_images = []
    for p in prompts:
        img = trainer.generate(p, num_images=1)
        all_images.append(img.cpu())
    all_images = torch.cat(all_images)
    vutils.save_image(all_images, os.path.join(args.output_dir, 'final_samples.png'), nrow=4)
    print(f"Saved final samples to {args.output_dir}/final_samples.png")


if __name__ == "__main__":
    main()
