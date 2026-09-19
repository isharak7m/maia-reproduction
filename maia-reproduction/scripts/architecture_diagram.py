import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('white')

# Colors
C_INPUT = '#E8F4FD'
C_CONV = '#B3D9F7'
C_RES = '#FFD699'
C_SE = '#FFB3B3'
C_POLICY = '#B3FFB3'
C_VALUE = '#D9B3FF'
C_OUTPUT = '#FFB3D9'

def draw_block(ax, x, y, w, h, color, label, fontsize=9, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='#333333', linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color='#1a1a1a')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

# Title
ax.text(7, 9.7, 'Maia-1100 Architecture', ha='center', va='center',
        fontsize=16, fontweight='bold', color='#1a1a1a')
ax.text(7, 9.35, 'Residual CNN with Squeeze-and-Excitation (18.6M params)',
        ha='center', va='center', fontsize=10, color='#555555')

# Input
draw_block(ax, 0.5, 7.5, 2.5, 1.2, C_INPUT, 'Input\n113 × 8 × 8', fontsize=10, bold=True)
ax.text(1.75, 7.2, '17 board + 96 history planes', ha='center', va='center',
        fontsize=7, color='#666666')

# Initial Conv
draw_block(ax, 4, 7.5, 2.5, 1.2, C_CONV, 'Conv 3×3\n113 → 256', fontsize=10, bold=True)
ax.text(5.25, 7.2, 'BatchNorm + ReLU', ha='center', va='center',
        fontsize=7, color='#666666')

# Arrow: Input -> Conv
draw_arrow(ax, 3.0, 8.1, 4.0, 8.1)

# Residual Blocks
draw_block(ax, 7.5, 7.5, 5.5, 1.2, C_RES, '×15 Residual Blocks (256 channels)', fontsize=10, bold=True)
ax.text(10.25, 7.2, 'Conv 3×3 → BN → ReLU → Conv 3×3 → BN + Skip', ha='center', va='center',
        fontsize=7, color='#666666')

# Arrow: Conv -> ResBlocks
draw_arrow(ax, 6.5, 8.1, 7.5, 8.1)

# SE Block
draw_block(ax, 7.5, 5.8, 5.5, 0.9, C_SE, 'Squeeze-and-Excitation (reduction=16)', fontsize=9, bold=True)
ax.text(10.25, 5.5, 'GlobalAvgPool → FC64 → ReLU → FC256 → Sigmoid → Scale', ha='center', va='center',
        fontsize=7, color='#666666')

# Arrow: ResBlocks -> SE
draw_arrow(ax, 10.25, 7.5, 10.25, 6.7)

# Policy Head
draw_block(ax, 1.5, 4.2, 3.5, 1.0, C_POLICY, 'Policy Head', fontsize=10, bold=True)
ax.text(3.25, 3.85, 'Conv 256→80→73 → Flatten → 4672 logits', ha='center', va='center',
        fontsize=7, color='#666666')

# Value Head
draw_block(ax, 9, 4.2, 3.5, 1.0, C_VALUE, 'Value Head', fontsize=10, bold=True)
ax.text(10.75, 3.85, 'Conv → FC256 → FC3 (win/draw/loss)', ha='center', va='center',
        fontsize=7, color='#666666')

# Arrows: SE -> Heads
draw_arrow(ax, 8.5, 5.8, 3.25, 5.2)
draw_arrow(ax, 12, 5.8, 10.75, 5.2)

# Output: Policy
draw_block(ax, 1.5, 2.5, 3.5, 0.9, C_OUTPUT, 'Move Prediction\n4672 move logits', fontsize=9, bold=True)
draw_arrow(ax, 3.25, 4.2, 3.25, 3.4)

# Output: Value
draw_block(ax, 9, 2.5, 3.5, 0.9, C_OUTPUT, 'Outcome Prediction\nWin / Draw / Loss', fontsize=9, bold=True)
draw_arrow(ax, 10.75, 4.2, 10.75, 3.4)

# Legend
legend_y = 0.8
legend_items = [
    (C_INPUT, 'Input'), (C_CONV, 'Convolution'), (C_RES, 'Residual Block'),
    (C_SE, 'SE Block'), (C_POLICY, 'Policy Head'), (C_VALUE, 'Value Head'),
    (C_OUTPUT, 'Output')
]
for i, (color, label) in enumerate(legend_items):
    x = 0.5 + i * 1.9
    box = FancyBboxPatch((x, legend_y), 0.4, 0.3, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor='#333333', linewidth=1)
    ax.add_patch(box)
    ax.text(x + 0.55, legend_y + 0.15, label, ha='left', va='center', fontsize=7.5, color='#333333')

plt.tight_layout()
plt.savefig('reports/architecture_diagram.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved reports/architecture_diagram.png")
