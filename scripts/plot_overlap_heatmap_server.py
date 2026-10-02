#!/usr/bin/env python3
"""10 方法间共同污染样本数热力图 (服务器版, 从 overlap_matrix.csv 读取, 矢量 PDF)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 数据文件
CSV_PATH = str(BASE_DIR / "data" / "chapter3_probes_v2" / "overlap_matrix.csv")
OUT_PNG = str(BASE_DIR / "figures" / "fig_overlap_heatmap.png")
OUT_PDF = str(BASE_DIR / "figures" / "fig_overlap_heatmap.pdf")

# 统一用论文里的方法名 (按 CSV 行序)
methods = ["Zlib", "Min-K%", "Min-K%++", "DC-PDD", "PPL",
           "PETAL", "Neighbor", "Reference", "Overlap", "CDD"]

# 读 CSV (第一列是 index)
df = pd.read_csv(CSV_PATH, index_col=0)
overlap = df.values.astype(int)
print(f"读取矩阵: {overlap.shape}")
print(f"对角线(各方法总数): {np.diag(overlap)}")

# 学术字体 + 矢量 PDF (服务器上用 STIXGeneral)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 6.5,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
})

n = len(methods)
fig, ax = plt.subplots(figsize=(3.5, 3.5))  # IEEE 单栏宽度 3.5 英寸

cmap = plt.cm.Blues
cmap.set_bad('white')  # 屏蔽区域显示白色

# 屏蔽右上角(上三角不含对角线), 只显示左下角
mask = np.triu(np.ones_like(overlap, dtype=bool), k=1)
overlap_masked = np.ma.masked_where(mask, overlap)

im = ax.imshow(overlap_masked, cmap=cmap, vmin=0, vmax=overlap.max(), aspect='equal')

# 数字: 只标下三角(含对角线), 深色白字/浅色黑字
for i in range(n):
    for j in range(i + 1):
        val = overlap[i, j]
        norm_val = val / overlap.max()
        text_color = 'white' if norm_val > 0.5 else 'black'
        ax.text(j, i, f'{val}', ha='center', va='center',
                fontsize=5.5, color=text_color)

ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.set_yticklabels(methods)
ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(n - 0.5, -0.5)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Co-contaminated samples', fontsize=8)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
print(f"Saved -> {OUT_PNG}")
print(f"Saved -> {OUT_PDF}")
