"""Generate warmup ablation figure for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from real evaluations
# Warmup=200 (v10_warmup200)
warmup200_steps = [20, 25, 30, 35]
warmup200_fid = [150.57, 107.94, 112.90, 152.35]

# Warmup=1000 (v6)
warmup1000_steps = [20, 30, 35]
warmup1000_fid = [134.00, 117.90, 82.64]

# Warmup=2000 (v10_warmup2000)
warmup2000_steps = [15, 20, 25, 30, 35]
warmup2000_fid = [166.49, 186.98, 180.24, 90.32, 55.45]

fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))

# Plot trajectories
ax.plot(warmup200_steps, warmup200_fid, 'o-', color='#e74c3c', label='Warmup = 200',
        linewidth=1.8, markersize=5)
ax.plot(warmup1000_steps, warmup1000_fid, 's-', color='#3498db', label='Warmup = 1,000',
        linewidth=1.8, markersize=5)
ax.plot(warmup2000_steps, warmup2000_fid, 'D-', color='#2ecc71', label='Warmup = 2,000',
        linewidth=1.8, markersize=5)

# Mark best points with stars
best_200_idx = np.argmin(warmup200_fid)
best_1000_idx = np.argmin(warmup1000_fid)
best_2000_idx = np.argmin(warmup2000_fid)

ax.plot(warmup200_steps[best_200_idx], warmup200_fid[best_200_idx], '*',
        color='#e74c3c', markersize=14, zorder=5)
ax.plot(warmup1000_steps[best_1000_idx], warmup1000_fid[best_1000_idx], '*',
        color='#3498db', markersize=14, zorder=5)
ax.plot(warmup2000_steps[best_2000_idx], warmup2000_fid[best_2000_idx], '*',
        color='#2ecc71', markersize=14, zorder=5)

# Annotate best FIDs
ax.annotate(f'{warmup200_fid[best_200_idx]:.1f}',
            (warmup200_steps[best_200_idx], warmup200_fid[best_200_idx]),
            textcoords="offset points", xytext=(8, 5), fontsize=8, color='#e74c3c')
ax.annotate(f'{warmup1000_fid[best_1000_idx]:.1f}',
            (warmup1000_steps[best_1000_idx], warmup1000_fid[best_1000_idx]),
            textcoords="offset points", xytext=(8, 5), fontsize=8, color='#3498db')
ax.annotate(f'{warmup2000_fid[best_2000_idx]:.1f}',
            (warmup2000_steps[best_2000_idx], warmup2000_fid[best_2000_idx]),
            textcoords="offset points", xytext=(8, 5), fontsize=8, color='#2ecc71')

ax.set_xlabel('Training Steps (K)', fontsize=10)
ax.set_ylabel('FID', fontsize=10)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(40, 200)
ax.set_xlim(13, 37)

plt.tight_layout()
plt.savefig('figures/warmup_ablation.pdf', dpi=300, bbox_inches='tight')
print("Saved figures/warmup_ablation.pdf")
