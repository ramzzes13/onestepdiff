"""Run ablation studies for AdaDMD on CIFAR-10.

Ablations:
1. LoRA rank: 4, 8, 16, 32, rank-warming (4->32)
2. Adaptive vs fixed weighting
3. Regularization strategy (DR only, LPIPS only, hybrid)
4. Density ratio network size
"""

import os
import sys
import json
import time
import copy
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adadmd.configs.cifar10 import CIFAR10Config
from adadmd.training.trainer import AdaDMDTrainer
from adadmd.evaluation.fid import compute_fid


def get_dataloader(batch_size=32):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                     num_workers=4, pin_memory=True, drop_last=True)


def get_eval_images(num_samples=5000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    images = []
    for imgs, _ in loader:
        images.append(imgs)
        if sum(i.shape[0] for i in images) >= num_samples:
            break
    return torch.cat(images)[:num_samples]


def run_single_ablation(config, name, base_model_path, num_iters=10000, device="cuda:0"):
    """Run a single ablation experiment."""
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"{'='*60}")

    config.output_dir = f"outputs/ablations/{name}"
    config.num_iterations = num_iters
    config.gpu_id = int(device.split(':')[-1])
    config.device = device

    os.makedirs(config.output_dir, exist_ok=True)

    trainer = AdaDMDTrainer(config)

    # Load base model
    if os.path.exists(base_model_path):
        trainer.base_model.load_state_dict(
            torch.load(base_model_path, map_location=trainer.device, weights_only=True)
        )
        for p in trainer.base_model.parameters():
            p.requires_grad_(False)
        trainer.generator.unet = copy.deepcopy(trainer.base_model)
        from adadmd.utils.ema import EMA
        trainer.ema_generator = EMA(trainer.generator, decay=config.ema_decay)

    dataloader = get_dataloader(config.batch_size)
    start_time = time.time()
    trainer.train(dataloader, num_iterations=num_iters)
    train_time = time.time() - start_time

    # Evaluate
    real_images = get_eval_images(config.num_eval_samples)
    fake_images = trainer.generate_samples(config.num_eval_samples)
    fid = compute_fid(real_images, fake_images, device=device, batch_size=64)

    result = {
        'name': name,
        'fid': fid,
        'train_time_seconds': train_time,
        'num_iterations': num_iters,
        'lora_rank_final': trainer.fake_score_model.rank,
        'config': {k: str(v) for k, v in vars(config).items()},
    }

    with open(os.path.join(config.output_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResult: FID = {fid:.2f} (time: {train_time:.0f}s)")
    return result


def ablation_lora_rank(base_model_path, device="cuda:0", num_iters=10000):
    """Ablation 1: LoRA rank."""
    results = []
    for rank in [4, 8, 16, 32]:
        config = CIFAR10Config()
        config.lora_rank_init = rank
        config.lora_rank_max = rank  # No warming for this ablation
        config.rank_warm_interval = 0
        result = run_single_ablation(
            config, f"lora_rank_{rank}", base_model_path,
            num_iters=num_iters, device=device
        )
        results.append(result)

    # Rank warming
    config = CIFAR10Config()
    config.lora_rank_init = 4
    config.lora_rank_max = 32
    config.rank_warm_interval = num_iters // 4
    result = run_single_ablation(
        config, "lora_rank_warming_4_to_32", base_model_path,
        num_iters=num_iters, device=device
    )
    results.append(result)

    return results


def ablation_weighting(base_model_path, device="cuda:0", num_iters=10000):
    """Ablation 2: Adaptive vs fixed weighting."""
    results = []

    # Adaptive (default)
    config = CIFAR10Config()
    result = run_single_ablation(
        config, "weight_adaptive", base_model_path,
        num_iters=num_iters, device=device
    )
    results.append(result)

    # Fixed (no density-ratio weighting)
    config = CIFAR10Config()
    config.lambda_dr = 0.0  # Disable DR
    result = run_single_ablation(
        config, "weight_fixed", base_model_path,
        num_iters=num_iters, device=device
    )
    results.append(result)

    return results


def ablation_regularization(base_model_path, device="cuda:0", num_iters=10000):
    """Ablation 3: Regularization strategy."""
    results = []

    # DR only
    config = CIFAR10Config()
    config.use_lpips = False
    config.lambda_lpips = 0.0
    config.lambda_dr = 0.5
    result = run_single_ablation(
        config, "reg_dr_only", base_model_path,
        num_iters=num_iters, device=device
    )
    results.append(result)

    # Hybrid (reduced LPIPS + DR)
    config = CIFAR10Config()
    config.use_lpips = False  # L1 proxy on CIFAR-10
    config.lambda_lpips = 0.1
    config.lambda_dr = 0.5
    result = run_single_ablation(
        config, "reg_hybrid", base_model_path,
        num_iters=num_iters, device=device
    )
    results.append(result)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--base_model", type=str, default="outputs/pretrain_cifar10/ddpm_final.pt")
    parser.add_argument("--num_iters", type=int, default=10000)
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["all", "rank", "weighting", "regularization"])
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    all_results = {}

    if args.ablation in ["all", "rank"]:
        print("\n" + "="*80)
        print("ABLATION 1: LoRA Rank")
        print("="*80)
        all_results["lora_rank"] = ablation_lora_rank(
            args.base_model, device=device, num_iters=args.num_iters
        )

    if args.ablation in ["all", "weighting"]:
        print("\n" + "="*80)
        print("ABLATION 2: Weighting Strategy")
        print("="*80)
        all_results["weighting"] = ablation_weighting(
            args.base_model, device=device, num_iters=args.num_iters
        )

    if args.ablation in ["all", "regularization"]:
        print("\n" + "="*80)
        print("ABLATION 3: Regularization Strategy")
        print("="*80)
        all_results["regularization"] = ablation_regularization(
            args.base_model, device=device, num_iters=args.num_iters
        )

    # Summary
    print("\n" + "="*80)
    print("ABLATION SUMMARY")
    print("="*80)
    for category, results in all_results.items():
        print(f"\n{category}:")
        for r in results:
            print(f"  {r['name']}: FID = {r['fid']:.2f} "
                  f"(rank={r['lora_rank_final']}, time={r['train_time_seconds']:.0f}s)")

    # Save all results
    os.makedirs("outputs/ablations", exist_ok=True)
    with open("outputs/ablations/all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
