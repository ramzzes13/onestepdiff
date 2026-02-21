"""Generate qualitative comparison figure: teacher vs student samples."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load images
teacher = np.array(Image.open('../outputs/cifar10_v5c/samples/teacher_targets.png'))
student_20k = np.array(Image.open('../outputs/cifar10_v5c/samples/step_20000.png'))
paired_20k = np.array(Image.open('../outputs/cifar10_v5c/samples/paired_20000.png'))

# Extract first 2 rows (16 images) from each 8x8 grid
# Each image is ~32px + 2px padding in the grid
# The grid images are 8 columns of 32x32 images
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
    [teacher, student_20k, paired_20k],
    ['DDPM Teacher (100-step)', 'AdaDMD Student (1-step, random noise)', 'AdaDMD Student (1-step, paired noise)']):
    # Show first 2 rows concatenated
    row0 = extract_row(img, 0)
    row1 = extract_row(img, 1)
    combined = np.concatenate([row0, row1], axis=0)
    ax.imshow(combined)
    ax.set_ylabel(title, fontsize=8, rotation=0, labelpad=160, va='center')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Qualitative Comparison (CIFAR-10 32×32, v5c at 20K steps)', fontsize=10, y=0.98)
plt.tight_layout(rect=[0.22, 0, 1, 0.96])
plt.savefig('figures/qualitative_comparison.pdf', dpi=200, bbox_inches='tight')
plt.savefig('figures/qualitative_comparison.png', dpi=200, bbox_inches='tight')
print("Saved figures/qualitative_comparison.{pdf,png}")
