"""Generate learning rate schedule figure showing phase alignment."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(6, 3))

# Cosine LR schedule parameters
total_steps = 35000
base_lr = 2e-5
min_lr_ratio = 0.1
min_lr = base_lr * min_lr_ratio

steps = np.arange(0, total_steps + 1)

# Different warmup durations
warmup_configs = [
    (1000, '#1f77b4', '1K warmup', '--', 0.7),
    (2000, '#2ca02c', '2K warmup', '--', 0.7),
    (5000, '#d62728', '5K warmup (phase-aligned)', '-', 1.0),
]

for warmup, color, label, ls, alpha in warmup_configs:
    lrs = []
    for s in steps:
        if s < warmup:
            lr = base_lr * s / warmup
        else:
            progress = (s - warmup) / (total_steps - warmup)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
        lrs.append(lr)
    ax.plot(steps / 1000, np.array(lrs) * 1e5, color=color, label=label,
            linestyle=ls, alpha=alpha, linewidth=1.5 if ls == '-' else 1.0)

# Mark Phase 1 / Phase 2 boundary
ax.axvline(x=5.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.text(5.1, 2.3, 'Phase 1|2\nboundary', fontsize=7, color='gray', va='top')

# Shade Phase 1
ax.axvspan(0, 5, alpha=0.05, color='blue')
ax.text(2.5, 0.15, 'Phase 1\n(regression only)', fontsize=7, ha='center',
        color='#555555', style='italic')

ax.set_xlabel('Training Step (K)', fontsize=9)
ax.set_ylabel('Learning Rate ($\\times 10^{-5}$)', fontsize=9)
ax.legend(fontsize=7, loc='upper right')
ax.set_xlim(0, 35)
ax.set_ylim(0, 2.5)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('figures/lr_schedule.pdf', dpi=150, bbox_inches='tight')
print("Saved figures/lr_schedule.pdf")
