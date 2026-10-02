#!/usr/bin/env python3
"""贪心前向选择折线图 - 论文发表级别"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ---- 数据: C+O 训练, 贪心从 DC-PDD 开始 ----
steps = list(range(1, 11))
labels = [
    "DC-PDD",
    "+Reference",
    "+Neighborhood",
    "+Min-K%",
    "+Zlib",
    "+CDD",
    "+PETAL",
    "+Min-K%++",
    "+Perplexity",
    "+Token\nOverlap",
]

avg_auc = [0.8462, 0.8716, 0.8884, 0.9178, 0.9269,
           0.9267, 0.9260, 0.9277, 0.9295, 0.9248]

codev  = [0.8005, 0.8491, 0.8547, 0.9019, 0.9173,
          0.9113, 0.9120, 0.9384, 0.9385, 0.9423]

rtl    = [0.8442, 0.8475, 0.8824, 0.9072, 0.9122,
          0.8996, 0.9001, 0.8711, 0.8753, 0.8658]

origen = [0.8937, 0.9183, 0.9280, 0.9443, 0.9512,
          0.9691, 0.9660, 0.9737, 0.9746, 0.9662]

# ---- 字体设置: 优先 STIX (matplotlib 内置, 学术风格) ----
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'CMU Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})

fig, ax = plt.subplots(figsize=(10, 4.8))

# 颜色 (colorblind-friendly + 学术配色)
c_avg = '#2166AC'
c_codev = '#B2182B'
c_rtl = '#4DAF4A'
c_ori = '#FF7F00'

x = np.arange(len(steps))

# 绘制折线
ax.plot(x, avg_auc, 'o-', color=c_avg, linewidth=2.4, markersize=8.5,
        markerfacecolor='white', markeredgewidth=2.2, label='Average', zorder=5)
ax.plot(x, codev, 's--', color=c_codev, linewidth=1.4, markersize=6.5,
        markerfacecolor='white', markeredgewidth=1.5, label='CodeV', zorder=3)
ax.plot(x, rtl, '^--', color=c_rtl, linewidth=1.4, markersize=6.5,
        markerfacecolor='white', markeredgewidth=1.5, label='RTLCoder', zorder=3)
ax.plot(x, origen, 'D--', color=c_ori, linewidth=1.4, markersize=6.5,
        markerfacecolor='white', markeredgewidth=1.5, label='OriGen', zorder=3)

# 最优 K=5 竖虚线 + 星标
opt_k = 4
ax.axvline(x=opt_k, color='#666666', linestyle=':', linewidth=1.0, alpha=0.6, zorder=1)
ax.plot(opt_k, avg_auc[opt_k], '*', color='#D41159', markersize=18,
        markeredgewidth=0.5, markeredgecolor='#444444', zorder=6)

# K=5 星标不加文字标注, 仅靠星号和竖虚线突出

# 坐标轴
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('AUC')
ax.set_xlabel('Method added (cumulative)')

# 纵轴
ax.set_ylim(0.83, 0.99)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# 网格
ax.grid(True, linestyle='--', linewidth=0.35, alpha=0.45, axis='y')
ax.set_axisbelow(True)

# 图例 — 放图内右上角, 避免和 x 轴标签冲突
ax.legend(ncol=4, loc='upper right',
          frameon=True, fancybox=False, edgecolor='#cccccc',
          framealpha=0.95, borderpad=0.5, handletextpad=0.5)

# 标题
# 标题在论文的 caption 中，图中不加标题

# Avg 数据点标注 — 自动避开星标
for i in range(len(steps)):
    if i == opt_k:
        offset = 0.007
    elif i == opt_k + 1 or i == opt_k - 1:
        offset = 0.005
    else:
        offset = 0.005
    ax.text(i, avg_auc[i] + offset, f'{avg_auc[i]:.3f}',
            ha='center', va='bottom', fontsize=8.2, color=c_avg,
            fontweight='bold')

plt.tight_layout()
out_dir = str(Path(__file__).resolve().parent.parent / 'figures')
plt.savefig(f'{out_dir}/fig_greedy_forward.pdf', format='pdf')
plt.savefig(f'{out_dir}/fig_greedy_forward.png', format='png')
print(f"Saved: {out_dir}/fig_greedy_forward.pdf  &  .png")
