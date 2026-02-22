"""Generate FID trajectory figure for the paper from real experiment data.
All numbers verified against outputs/eval_*/results.json files."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# FID data from real evaluations (verified)
# v5b: decaying regression weight (1.0 -> 0.3), constant LR
v5b_steps = [5, 10, 15, 20, 25, 30, 50]
v5b_fid =   [232.78, 140.16, 139.80, 139.77, 155.91, 174.83, 152.72]

# v5c: constant regression weight, constant LR
v5c_steps = [10, 15, 20, 25]
v5c_fid =   [180.72, 141.18, 115.01, 138.74]

# v6: constant regression weight, cosine LR, lambda_dm=5e-5
v6_steps = [10, 15, 20, 25, 30, 35]
v6_fid =   [178.42, 185.16, 134.00, 152.66, 117.90, 82.64]

# v7: constant regression weight, cosine LR, lambda_dm=0.001
v7_steps = [10, 15, 20, 25, 30, 35]
v7_fid =   [216.04, 158.65, 148.18, 151.36, 174.70, 107.85]

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.8))

ax.plot(v5b_steps, v5b_fid, 'o-', color='#2196F3', linewidth=1.5, markersize=5,
        label='Decay reg, const LR', alpha=0.8)
ax.plot(v5c_steps, v5c_fid, 's-', color='#FF9800', linewidth=1.5, markersize=5,
        label='Const reg, const LR', alpha=0.8)
ax.plot(v6_steps, v6_fid, 'D-', color='#4CAF50', linewidth=2.5, markersize=6,
        label=r'Cosine LR, $\lambda_{\mathrm{DM}}$=5e-5', alpha=0.9)
ax.plot(v7_steps, v7_fid, '^--', color='#9C27B0', linewidth=1.5, markersize=5,
        label=r'Cosine LR, $\lambda_{\mathrm{DM}}$=1e-3', alpha=0.7)

# Annotate key results
ax.annotate('82.64', xy=(35, 82.64), xytext=(31.5, 68),
            fontsize=9, fontweight='bold', color='#4CAF50',
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.2))
ax.annotate('115.01', xy=(20, 115.01), xytext=(14, 102),
            fontsize=8, color='#FF9800',
            arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1))

ax.set_xlabel('Training Steps (K)', fontsize=11)
ax.set_ylabel('FID', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(60, 240)
ax.set_xlim(3, 37)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper/figures/fid_trajectory.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/fid_trajectory.png', bbox_inches='tight', dpi=150)
print("Saved fid_trajectory.pdf and fid_trajectory.png")
