"""Generate schedule length ablation figure from real experiment data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Real data from experiments (verified against results.json files)
# v5c: 25K steps, constant LR
v5c_steps = [10, 15, 20, 25]
v5c_fid = [180.72, 141.18, 115.01, 138.74]

# v6: 35K steps, cosine LR
v6_steps = [10, 15, 20, 25, 30, 35]
v6_fid = [178.42, 185.16, 134.00, 152.66, 117.90, 82.64]

# v9: 40K steps, cosine LR
v9_steps = [20, 30, 35, 40]
v9_fid = [175.21, 103.01, 119.87, 125.86]

# v8: 50K steps, cosine LR
v8_steps = [20, 25, 30, 35, 40, 45, 50]
v8_fid = [106.52, 147.97, 139.59, 153.88, 101.96, 127.10, 161.93]

fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

# Plot each schedule
ax.plot(v5c_steps, v5c_fid, 'o--', color='#7570b3', label='25K, const LR',
        markersize=5, linewidth=1.5)
ax.plot(v6_steps, v6_fid, 's-', color='#1b9e77', label='35K, cosine',
        markersize=5, linewidth=2)
ax.plot(v9_steps, v9_fid, 'D-.', color='#d95f02', label='40K, cosine',
        markersize=5, linewidth=1.5)
ax.plot(v8_steps, v8_fid, '^:', color='#e7298a', label='50K, cosine',
        markersize=5, linewidth=1.5)

# Mark best points with stars
best_configs = [
    (20, 115.01, '#7570b3'),   # v5c best
    (35, 82.64, '#1b9e77'),    # v6 best
    (30, 103.01, '#d95f02'),   # v9 best
    (40, 101.96, '#e7298a'),   # v8 best
]
for step, fid, color in best_configs:
    ax.plot(step, fid, '*', color=color, markersize=14, markeredgecolor='black',
            markeredgewidth=0.5, zorder=5)

ax.set_xlabel('Training Steps (K)', fontsize=11)
ax.set_ylabel('FID', fontsize=11)
ax.set_xlim(8, 52)
ax.set_ylim(70, 200)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title('Schedule Length Ablation', fontsize=12)

plt.tight_layout()
plt.savefig('paper/figures/schedule_ablation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/schedule_ablation.png', dpi=150, bbox_inches='tight')
print("Saved schedule_ablation.pdf and .png")
