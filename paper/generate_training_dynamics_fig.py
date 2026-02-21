"""Generate training dynamics figure from v5b and v5c logs."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load v5b (decaying reg) and v5c (constant reg) training logs
with open('../outputs/cifar10_v5b/training_log.json') as f:
    log_v5b = json.load(f)

with open('../outputs/cifar10_v5c/training_log.json') as f:
    log_v5c = json.load(f)

# Extract data
def extract(log, key):
    steps = [x['step'] for x in log]
    vals = [x[key] for x in log]
    return steps, vals

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# (a) LPIPS Regression Loss
ax = axes[0, 0]
s_b, v_b = extract(log_v5b, 'reg')
s_c, v_c = extract(log_v5c, 'reg')
ax.plot(s_b, v_b, alpha=0.3, color='C0', linewidth=0.5)
ax.plot(s_c, v_c, alpha=0.3, color='C1', linewidth=0.5)
# Smoothed
window = 50
ax.plot(s_b[window:], np.convolve(v_b, np.ones(window)/window, mode='valid')[:len(s_b)-window],
        color='C0', label='Decaying reg (v5b)', linewidth=1.5)
ax.plot(s_c[window:], np.convolve(v_c, np.ones(window)/window, mode='valid')[:len(s_c)-window],
        color='C1', label='Constant reg (v5c)', linewidth=1.5)
ax.set_xlabel('Training Step')
ax.set_ylabel('LPIPS Loss')
ax.set_title('(a) Regression Loss')
ax.legend(fontsize=8)
ax.axvline(x=5000, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.text(5200, ax.get_ylim()[1]*0.95, 'Phase 2', fontsize=7, color='gray')
ax.grid(True, alpha=0.3)

# (b) NCE Accuracy
ax = axes[0, 1]
s_b, v_b = extract(log_v5b, 'acc')
s_c, v_c = extract(log_v5c, 'acc')
ax.plot(s_b[window:], np.convolve(v_b, np.ones(window)/window, mode='valid')[:len(s_b)-window],
        color='C0', label='Decaying reg (v5b)', linewidth=1.5)
ax.plot(s_c[window:], np.convolve(v_c, np.ones(window)/window, mode='valid')[:len(s_c)-window],
        color='C1', label='Constant reg (v5c)', linewidth=1.5)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.set_xlabel('Training Step')
ax.set_ylabel('NCE Accuracy')
ax.set_title('(b) Density-Ratio NCE Accuracy')
ax.legend(fontsize=8)
ax.axvline(x=5000, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.grid(True, alpha=0.3)

# (c) LoRA Denoising Loss
ax = axes[1, 0]
s_b, v_b = extract(log_v5b, 'lora')
s_c, v_c = extract(log_v5c, 'lora')
ax.plot(s_b[window:], np.convolve(v_b, np.ones(window)/window, mode='valid')[:len(s_b)-window],
        color='C0', label='Decaying reg (v5b)', linewidth=1.5)
ax.plot(s_c[window:], np.convolve(v_c, np.ones(window)/window, mode='valid')[:len(s_c)-window],
        color='C1', label='Constant reg (v5c)', linewidth=1.5)
ax.set_xlabel('Training Step')
ax.set_ylabel('LoRA Loss')
ax.set_title('(c) Fake Score LoRA Loss')
ax.legend(fontsize=8)
ax.axvline(x=5000, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.grid(True, alpha=0.3)

# (d) Output Standard Deviation
ax = axes[1, 1]
s_b, v_b = extract(log_v5b, 'x_std')
s_c, v_c = extract(log_v5c, 'x_std')
ax.plot(s_b[window:], np.convolve(v_b, np.ones(window)/window, mode='valid')[:len(s_b)-window],
        color='C0', label='Decaying reg (v5b)', linewidth=1.5)
ax.plot(s_c[window:], np.convolve(v_c, np.ones(window)/window, mode='valid')[:len(s_c)-window],
        color='C1', label='Constant reg (v5c)', linewidth=1.5)
ax.set_xlabel('Training Step')
ax.set_ylabel('Output Std')
ax.set_title('(d) Generator Output Diversity')
ax.legend(fontsize=8)
ax.axvline(x=5000, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/training_dynamics.pdf', dpi=150, bbox_inches='tight')
plt.savefig('figures/training_dynamics.png', dpi=150, bbox_inches='tight')
print("Saved figures/training_dynamics.{pdf,png}")
