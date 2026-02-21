"""Master script: Run the complete AdaDMD pipeline.

1. Pretrain base DDPM on CIFAR-10
2. Run AdaDMD distillation
3. Evaluate (FID, density-ratio verification)
4. Run ablation studies
5. Generate summary report
"""

import os
import sys
import json
import time
import subprocess


def run_command(cmd, description=""):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: Command failed with return code {result.returncode}")
    return result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--skip_distill", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_ablations", action="store_true")
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--distill_iters", type=int, default=50000)
    parser.add_argument("--ablation_iters", type=int, default=10000)
    args = parser.parse_args()

    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    start_time = time.time()

    # Step 1: Pretrain DDPM
    if not args.skip_pretrain:
        run_command(
            f"python3 adadmd/training/pretrain_ddpm.py "
            f"--epochs {args.pretrain_epochs} --batch_size 128 "
            f"--device cuda:0 --output_dir outputs/pretrain_cifar10",
            "Pretrain DDPM on CIFAR-10"
        )

    # Step 2: Run AdaDMD distillation
    if not args.skip_distill:
        run_command(
            f"python3 train_cifar10.py --gpu 0 "
            f"--pretrain_epochs 0 --distill_iters {args.distill_iters}",
            "AdaDMD Distillation"
        )

    # Step 3: Evaluate
    if not args.skip_eval:
        run_command(
            f"python3 evaluate_cifar10.py --gpu 0 --num_samples 5000",
            "Evaluate CIFAR-10 FID"
        )

    # Step 4: Ablations
    if not args.skip_ablations:
        run_command(
            f"python3 run_ablations.py --gpu 0 "
            f"--base_model outputs/pretrain_cifar10/ddpm_final.pt "
            f"--num_iters {args.ablation_iters}",
            "Run Ablation Studies"
        )

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE! Total time: {total_time/3600:.1f} hours")
    print(f"{'='*60}")

    # Generate summary
    generate_summary()


def generate_summary():
    """Generate a summary of all results."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    # CIFAR-10 main results
    eval_path = "outputs/cifar10/eval/results.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            results = json.load(f)
        print(f"\nCIFAR-10 Main Results:")
        print(f"  FID: {results.get('fid', 'N/A')}")
        for n in [1, 4, 8]:
            k = f'fid_verified_n{n}'
            if k in results:
                print(f"  FID (top-1 of {n}): {results[k]}")

    # Ablation results
    abl_path = "outputs/ablations/all_results.json"
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            ablations = json.load(f)
        print(f"\nAblation Results:")
        for category, results in ablations.items():
            print(f"\n  {category}:")
            for r in results:
                print(f"    {r['name']}: FID = {r.get('fid', 'N/A')}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
