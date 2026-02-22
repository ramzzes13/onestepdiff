"""Generate warmup ablation figure for the paper with all 5 warmup durations."""
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

# Warmup=3000 (v10_warmup3000)
warmup3000_steps = [25, 30, 35]
warmup3000_fid = [99.33, 123.23, 99.07]

# Warmup=5000 (v10_warmup5000, phase-aligned)
warmup5000_steps = [20, 25, 30, 35]
warmup5000_fid = [114.71, 50.05, 28.06, 29.81]

fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.4))

# Plot trajectories
ax.plot(warmup200_steps, warmup200_fid, 'o--', color='#e74c3c', label='w=200',
        linewidth=1.2, markersize=4, alpha=0.7)
ax.plot(warmup1000_steps, warmup1000_fid, 's--', color='#3498db', label='w=1,000',
        linewidth=1.2, markersize=4, alpha=0.7)
ax.plot(warmup2000_steps, warmup2000_fid, 'D--', color='#2ecc71', label='w=2,000',
        linewidth=1.5, markersize=4, alpha=0.8)
ax.plot(warmup3000_steps, warmup3000_fid, '^--', color='#9b59b6', label='w=3,000',
        linewidth=1.2, markersize=4, alpha=0.7)
ax.plot(warmup5000_steps, warmup5000_fid, 'P-', color='#e91e63', label='w=5,000 (Phase 1)',
        linewidth=2.5, markersize=7, alpha=0.95)

# Mark best points with stars
for steps, fids, color in [
    (warmup200_steps, warmup200_fid, '#e74c3c'),
    (warmup1000_steps, warmup1000_fid, '#3498db'),
    (warmup2000_steps, warmup2000_fid, '#2ecc71'),
    (warmup3000_steps, warmup3000_fid, '#9b59b6'),
    (warmup5000_steps, warmup5000_fid, '#e91e63'),
]:
    best_idx = np.argmin(fids)
    ax.plot(steps[best_idx], fids[best_idx], '*', color=color, markersize=14, zorder=5)

# Annotate the best result
ax.annotate('28.06', (30, 28.06), textcoords="offset points", xytext=(5, 8),
            fontsize=9, fontweight='bold', color='#e91e63')

ax.set_xlabel('Training Steps (K)', fontsize=10)
ax.set_ylabel('FID', fontsize=10)
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(20, 200)
ax.set_xlim(13, 37)

plt.tight_layout()
plt.savefig('figures/warmup_ablation.pdf', dpi=300, bbox_inches='tight')
print("Saved figures/warmup_ablation.pdf")
