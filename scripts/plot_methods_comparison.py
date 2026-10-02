#!/usr/bin/env python3
"""各方法 AUC + F1 分组柱状图 (矢量 PDF, 学术风格)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 数据 (按 AUC 降序) —— 来自 final_vs_baselines_5f.py 重跑 (C+O, 5f, none)
methods = ["Boost", "DC-PDD", "Min-K%", "PPL", "PETAL",
           "Min-K%++", "Reference", "Zlib", "CDD", "Neighborhood", "Overlap"]
auc = [0.9269, 0.8499, 0.7912, 0.7854, 0.7612,
       0.6633, 0.6526, 0.6314, 0.6167, 0.6064, 0.5876]
f1  = [0.8720, 0.6645, 0.7598, 0.7536, 0.6599,
       0.7032, 0.7043, 0.6999, 0.4457, 0.7043, 0.6578]

# 学术字体 (矢量 PDF: 内嵌 TrueType, 无 Type3)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,   # 内嵌 TrueType, 避免 Type 3
    'ps.fonttype': 42,
    'font.size': 7,
    'axes.labelsize': 7,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})

x = np.arange(len(methods))
bar_width = 0.45

# AUC 蓝, F1 橙
auc_color = '#3A7BBF'
f1_color  = '#E8833A'

fig, ax = plt.subplots(figsize=(3.5, 2.6))  # IEEE 单栏宽度 3.5 英寸

# 柱子: align=edge 明确左边缘, 黑色边框, AUC 左边缘 x-bar_width, F1 左边缘 x
bars1 = ax.bar(x - bar_width, auc, bar_width, align='edge', label='AUC',
               color=auc_color, edgecolor='black', linewidth=0.6, zorder=3)
bars2 = ax.bar(x, f1, bar_width, align='edge', label='F1',
               color=f1_color, edgecolor='black', linewidth=0.6, zorder=3)

# 数值标注: 柱子正中心上方, 逆时针旋转 90 度, 略窄于柱宽
for xi, a in zip(x, auc):
    ax.text(xi - bar_width/2, a + 0.08, f'{a:.3f}', ha='center', va='center', fontsize=6, rotation=90)
for xi, f in zip(x, f1):
    ax.text(xi + bar_width/2, f + 0.08, f'{f:.3f}', ha='center', va='center', fontsize=6, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.10)
ax.set_yticks(np.arange(0, 1.01, 0.1))

# 背景平行虚线 (在柱状图图层下方)
ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# 上下左右都用黑色直线包起来
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(True)
    ax.spines[s].set_linewidth(0.8)
    ax.spines[s].set_color('black')

# 图例: 柱状图内右上角, 上下排列, 无灰框
ax.legend(ncol=1, loc='upper right', frameon=False, columnspacing=1.0)

out = str(Path(__file__).resolve().parent.parent / "figures" / "fig_methods_comparison")
plt.savefig(out + ".png", dpi=300, bbox_inches="tight")
plt.savefig(out + ".pdf", bbox_inches="tight")
print(f"Saved -> {out}.png / .pdf")
