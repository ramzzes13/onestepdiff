"""Generate qualitative comparison figure: teacher vs best student (warmup=5000, 30K steps)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load images from warmup=5000 model at 30K steps (FID 28.06)
teacher = np.array(Image.open('../outputs/cifar10_v10_warmup5000/samples/teacher_targets.png'))
student_30k = np.array(Image.open('../outputs/cifar10_v10_warmup5000/samples/step_30000.png'))
paired_30k = np.array(Image.open('../outputs/cifar10_v10_warmup5000/samples/paired_30000.png'))

h, w, c = teacher.shape
cell_h = h // 8
cell_w = w // 8

def extract_row(grid, row_idx, ncols=8):
    """Extract a single row of images from a grid."""
    cells = []
    for col in range(ncols):
        y0 = row_idx * cell_h
        x0 = col * cell_w
        cells.append(grid[y0:y0+cell_h, x0:x0+cell_w])
    return np.concatenate(cells, axis=1)

fig, axes = plt.subplots(3, 1, figsize=(8, 3.6))

for ax, img, title in zip(axes,
    [teacher, student_30k, paired_30k],
    ['DDPM Teacher (100-step)', 'AdaDMD Student (1-step, random noise)', 'AdaDMD Student (1-step, paired noise)']):
    row0 = extract_row(img, 0)
    row1 = extract_row(img, 1)
    combined = np.concatenate([row0, row1], axis=0)
    ax.imshow(combined)
    ax.set_ylabel(title, fontsize=8, rotation=0, labelpad=160, va='center')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Best Model: Phase-Aligned Warmup at 30K steps (FID 28.06)', fontsize=10, y=0.98)
plt.tight_layout(rect=[0.22, 0, 1, 0.96])
plt.savefig('figures/qualitative_comparison.pdf', dpi=200, bbox_inches='tight')
plt.savefig('figures/qualitative_comparison.png', dpi=200, bbox_inches='tight')
print("Saved figures/qualitative_comparison.{pdf,png}")
