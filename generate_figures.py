"""Generate paper figures from experimental data."""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

FIG_DIR = '/home/mekashirskiy/rom4ik/onestepdiff/paper/figures'
os.makedirs(FIG_DIR, exist_ok=True)


def fig_lora_rank_ablation():
    """Figure 1: LoRA rank vs FID."""
    ranks = [4, 8, 16, 32]
    fids = [224.98, 223.59, 226.03, 227.79]
    warming_fid = 224.40

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(ranks, fids, 'o-', color='#2196F3', linewidth=2, markersize=8,
            label='Fixed rank')
    ax.axhline(y=warming_fid, color='#FF5722', linestyle='--', linewidth=1.5,
               label=f'Rank warming 4→32 ({warming_fid:.1f})')
    ax.set_xlabel('LoRA Rank')
    ax.set_ylabel('FID ↓')
    ax.set_xticks(ranks)
    ax.legend()
    ax.set_ylim(222, 229)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'lora_rank_ablation.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'lora_rank_ablation.png'), bbox_inches='tight')
    plt.close(fig)
    print("Generated: lora_rank_ablation")


def fig_weighting_comparison():
    """Figure 2: Adaptive vs fixed weighting."""
    strategies = ['Fixed\n(σ²/α)', 'Adaptive\n(DR)']
    fids = [225.97, 223.87]
    colors = ['#90CAF9', '#2196F3']

    fig, ax = plt.subplots(figsize=(3.5, 3))
    bars = ax.bar(strategies, fids, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
    ax.set_ylabel('FID ↓')
    ax.set_ylim(222, 227)
    for bar, fid in zip(bars, fids):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{fid:.2f}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'weighting_comparison.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'weighting_comparison.png'), bbox_inches='tight')
    plt.close(fig)
    print("Generated: weighting_comparison")


def fig_regularization_ablation():
    """Figure 3: Regularization strategy comparison."""
    strategies = ['DR only', 'Hybrid\n(LPIPS+DR)']
    fids = [223.45, 226.56]
    colors = ['#4CAF50', '#FFC107']

    fig, ax = plt.subplots(figsize=(3.5, 3))
    bars = ax.bar(strategies, fids, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
    ax.set_ylabel('FID ↓')
    ax.set_ylim(222, 228)
    for bar, fid in zip(bars, fids):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{fid:.2f}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'regularization_ablation.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'regularization_ablation.png'), bbox_inches='tight')
    plt.close(fig)
    print("Generated: regularization_ablation")


def fig_dr_verification():
    """Figure 4: DR verification improvement."""
    models = ['SmallUNet\n(MSE, 20K)', 'HF DDPM\n(MSE, 5K)', 'HF DDPM\n(LPIPS, 20K)']
    n1 = [222.12, 237.09, 139.77]
    n4 = [219.93, 234.23, 135.22]
    n8 = [220.10, 233.93, 137.38]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars1 = ax.bar(x - width, n1, width, label='N=1 (baseline)', color='#E0E0E0',
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, n4, width, label='N=4', color='#64B5F6',
                   edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, n8, width, label='N=8', color='#1976D2',
                   edgecolor='black', linewidth=0.5)

    ax.set_ylabel('FID ↓')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(130, 245)
    ax.grid(True, axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'dr_verification.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'dr_verification.png'), bbox_inches='tight')
    plt.close(fig)
    print("Generated: dr_verification")


def fig_fid_trajectory():
    """Figure 7: FID trajectory during training."""
    steps = [5, 10, 15, 20]
    fids = [232.78, 140.16, 139.80, 139.77]
    fids_dr4 = [233.53, 140.28, 141.84, 135.22]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(steps, fids, 'o-', color='#2196F3', linewidth=2, markersize=8,
            label='Standard')
    ax.plot(steps, fids_dr4, 's--', color='#FF5722', linewidth=2, markersize=7,
            label='DR-verified (N=4)')
    ax.axvline(x=5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(5.2, 200, 'Phase 2\nstarts', fontsize=8, color='gray')
    ax.set_xlabel('Training Steps (K)')
    ax.set_ylabel('FID ↓')
    ax.legend()
    ax.set_ylim(125, 245)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fid_trajectory.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fid_trajectory.png'), bbox_inches='tight')
    plt.close(fig)
    print("Generated: fid_trajectory")


def fig_memory_comparison():
    """Figure 5: Memory comparison."""
    components = ['Base\n(frozen)', 'Generator', 'Fake Score\n(LoRA)', 'DR Net', 'LPIPS', 'Optim\n+buffers']
    lora_mem = [147, 147, 4, 1.3, 9, 670]
    full_mem = [147, 147, 147, 1.3, 9, 900]

    x = np.arange(len(components))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - width/2, full_mem, width, label='Full copy (DMD)', color='#FFCDD2',
           edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, lora_mem, width, label='LoRA (AdaDMD)', color='#C8E6C9',
           edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Memory (MB)')
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'memory_comparison.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'memory_comparison.png'), bbox_inches='tight')
    plt.close(fig)
    print("Generated: memory_comparison")


def fig_combined_ablation():
    """Figure 6: Combined ablation summary."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Panel A: LoRA rank
    ranks = [4, 8, 16, 32]
    fids = [224.98, 223.59, 226.03, 227.79]
    axes[0].plot(ranks, fids, 'o-', color='#2196F3', linewidth=2, markersize=8)
    axes[0].axhline(y=224.40, color='#FF5722', linestyle='--', linewidth=1.5,
                    label=f'Warming 4→32')
    axes[0].set_xlabel('LoRA Rank')
    axes[0].set_ylabel('FID ↓')
    axes[0].set_xticks(ranks)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(222, 229)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('(a) LoRA Rank')

    # Panel B: Weighting
    strats = ['Fixed', 'Adaptive']
    wfids = [225.97, 223.87]
    colors = ['#90CAF9', '#2196F3']
    axes[1].bar(strats, wfids, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
    axes[1].set_ylabel('FID ↓')
    axes[1].set_ylim(222, 227)
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].set_title('(b) Weighting')
    for i, (s, f) in enumerate(zip(strats, wfids)):
        axes[1].text(i, f + 0.1, f'{f:.1f}', ha='center', fontsize=8)

    # Panel C: Regularization
    rstrats = ['DR only', 'Hybrid']
    rfids = [223.45, 226.56]
    rcolors = ['#4CAF50', '#FFC107']
    axes[2].bar(rstrats, rfids, color=rcolors, edgecolor='black', linewidth=0.5, width=0.5)
    axes[2].set_ylabel('FID ↓')
    axes[2].set_ylim(222, 228)
    axes[2].grid(True, axis='y', alpha=0.3)
    axes[2].set_title('(c) Regularization')
    for i, (s, f) in enumerate(zip(rstrats, rfids)):
        axes[2].text(i, f + 0.1, f'{f:.1f}', ha='center', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'combined_ablation.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'combined_ablation.png'), bbox_inches='tight')
    plt.close(fig)
    print("Generated: combined_ablation")


if __name__ == '__main__':
    fig_lora_rank_ablation()
    fig_weighting_comparison()
    fig_regularization_ablation()
    fig_dr_verification()
    fig_memory_comparison()
    fig_combined_ablation()
    fig_fid_trajectory()
    print(f"\nAll figures saved to {FIG_DIR}/")
