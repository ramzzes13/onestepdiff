#!/bin/bash
# Evaluate warmup ablation experiments at key checkpoints
# Run after training completes

set -e

# Warmup=200 evaluations
for step in 15000 20000 25000 30000 35000; do
    ckpt="outputs/cifar10_v10_warmup200/checkpoints/step_${step}.pt"
    outdir="outputs/eval_v10_warmup200_${step}"
    if [ -f "$ckpt" ] && [ ! -f "$outdir/results.json" ]; then
        echo "Evaluating warmup=200 at step $step..."
        CUDA_VISIBLE_DEVICES=0 python evaluate_fid_v5.py \
            --checkpoint "$ckpt" \
            --output_dir "$outdir" \
            --gpu 0 --num_samples 5000
    fi
done

# Warmup=2000 evaluations
for step in 15000 20000 25000 30000 35000; do
    ckpt="outputs/cifar10_v10_warmup2000/checkpoints/step_${step}.pt"
    outdir="outputs/eval_v10_warmup2000_${step}"
    if [ -f "$ckpt" ] && [ ! -f "$outdir/results.json" ]; then
        echo "Evaluating warmup=2000 at step $step..."
        CUDA_VISIBLE_DEVICES=1 python evaluate_fid_v5.py \
            --checkpoint "$ckpt" \
            --output_dir "$outdir" \
            --gpu 0 --num_samples 5000
    fi
done

echo "All evaluations complete!"
