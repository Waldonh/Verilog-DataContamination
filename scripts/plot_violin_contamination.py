#!/usr/bin/env python3
"""9 个 model 的污染概率 violin plot (样本级污染强度谱系, seaborn 精致版)"""
import json, hashlib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# 最优 5f
OPTIMAL_5F = ["DCPDD", "Reference", "neighbor", "mink_0.2", "zlib"]

SCORER_MAP = {
    "mink_family": ["zlib", "mink_0.2", "mink++_0.2"],
    "dcpdd_ppl": ["DCPDD", "ppl"],
    "petal": ["PETAL"],
    "neighbor": ["neighbor"],
    "reference": ["Reference"],
    "token_overlap": ["Token_Overlap"],
    "cdd": ["CDD"],
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FEAT_DIR = DATA_DIR / "features" / "golden"         # CodeV/OriGen 特征
TRAIN_GOLDEN_DIR = DATA_DIR / "golden"                     # golden 数据集
FEAT_ROOT = DATA_DIR / "features" / "benchmark"            # benchmark 特征

# 9 个 model: (显示名, 特征目录名)
MODELS = [
    ("CodeGen-6B-Multi",     "CodeGen-6B-Multi"),
    ("Phi-2",                "Phi-2"),
    ("DeepSeek-Coder-6.7B",  "DeepSeek-Coder-6.7B"),
    ("Qwen2.5-Coder",        "Qwen2.5-Coder-7B"),
    ("ChipGPT",              "ChipGPT-Llama2-7B"),
    ("RTLCoder",             "RTLCoder"),
    ("OriGen",               "OriGen"),
    ("VeriCoder",            "VeriCoder"),
    ("VeriGen",              "FineTuned-CodeGen-6B"),
]

BENCHMARKS = ["VerilogEval_v1", "VerilogEval_v2", "RTLLM_v1", "RTLLM_v2",
              "RealBench_module", "RealBench_system"]

OUT_PNG = str(BASE_DIR / "figures" / "fig_violin_contamination.png")
OUT_PDF = str(BASE_DIR / "figures" / "fig_violin_contamination.pdf")


def code_hash(c):
    return hashlib.sha256(c.encode("utf-8")).hexdigest()


def load_features(feat_dir):
    feat_dir = Path(feat_dir); combined = {}; labels = None
    for subd, cols in SCORER_MAP.items():
        path = feat_dir / subd / "results.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if labels is None:
            labels = np.array([int(r.get("label", 0)) for r in rows])
        for col in cols:
            actual = {"Token_Overlap": "contamination_score", "CDD": "contamination_score",
                      "Reference": "ref", "ppl": "PPL"}.get(col, col)
            combined[col] = [float(r.get(actual, np.nan)) for r in rows[:len(labels)]]
    if not combined:
        return None, None
    min_len = min(len(v) for v in combined.values())
    return pd.DataFrame({k: v[:min_len] for k, v in combined.items()}), labels[:min_len]


# ===== 1. 训练 CatBoost-5f (C+O) =====
print("训练 CatBoost-5f (C+O)...")
train_feats = {}; train_labels = {}
for m in ["CodeV", "OriGen"]:
    df, lb = load_features(TRAIN_FEAT_DIR / m)
    train_feats[m] = df; train_labels[m] = lb

splits = {}
for m in ["CodeV", "OriGen"]:
    all_path = TRAIN_FEAT_DIR / f"{m}_all.jsonl"
    all_rows = [json.loads(l) for l in open(all_path, encoding="utf-8") if l.strip()]
    all_hashes = [code_hash(r["Response"]) for r in all_rows]
    test_path = TRAIN_GOLDEN_DIR / f"{m}_test_200.jsonl"
    test_hashes = {code_hash(json.loads(l)["Response"]) for l in open(test_path, encoding="utf-8") if l.strip()}
    te_idx = [i for i, h in enumerate(all_hashes) if h in test_hashes]
    tr_idx = [i for i in range(len(all_hashes)) if i not in set(te_idx)]
    splits[m] = {"train": tr_idx, "test": te_idx}

X_train = pd.concat([train_feats["CodeV"].iloc[splits["CodeV"]["train"]],
                     train_feats["OriGen"].iloc[splits["OriGen"]["train"]]],
                    ignore_index=True)[OPTIMAL_5F].astype(float)
y_train = np.concatenate([train_labels["CodeV"][splits["CodeV"]["train"]],
                          train_labels["OriGen"][splits["OriGen"]["train"]]])

model = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.03,
                           loss_function="Logloss", eval_metric="AUC",
                           random_seed=42, verbose=False, allow_writing_files=False)
model.fit(X_train, y_train)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X_train))
for tr_idx, val_idx in skf.split(X_train, y_train):
    m = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.03,
                           loss_function="Logloss", eval_metric="AUC",
                           random_seed=42, verbose=False, allow_writing_files=False)
    m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
    oof[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]
best_t = 0.5; best_f = -1.0
for t in np.arange(0.05, 0.951, 0.005):
    f = f1_score(y_train, (oof >= t).astype(int))
    if f > best_f: best_t, best_f = float(t), f
print(f"OOF 阈值 τ* = {best_t:.4f} (OOF F1={best_f:.4f})")

# ===== 2. 计算每个样本的 prob =====
print("\n计算 benchmark 样本的污染概率...")
probs_by_model = {}
for display_name, dir_name in MODELS:
    all_probs = []
    for bname in BENCHMARKS:
        feat_dir = FEAT_ROOT / dir_name / bname
        df, _ = load_features(feat_dir)
        if df is None or len(df) == 0:
            continue
        X = df[OPTIMAL_5F].astype(float).fillna(0.0)
        prob = model.predict_proba(X)[:, 1]
        all_probs.extend(prob.tolist())
    probs_by_model[display_name] = np.array(all_probs)
    print(f"  {display_name}: {len(all_probs)} 样本, mean p={np.mean(all_probs):.3f}")

# ===== 3. Ridgeline plot (山脊图) =====
from scipy.stats import gaussian_kde

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

fig, ax = plt.subplots(figsize=(3.5, 3.6))  # IEEE 单栏宽度, 高度放大

model_names = [d for d, _ in MODELS]
violin_data = [probs_by_model[d] for d in model_names]

# 9 种颜色: CodeGen/Phi-2 保持 husl 原色, 其余 7 个换高区分度色 (避开红橙)
colors = sns.color_palette('husl', len(MODELS))
QUAL_COLORS = {
    2: "#2a78d6",  # DeepSeek-Coder-6.7B (blue)
    3: "#1baf7a",  # Qwen2.5-Coder       (teal)
    4: "#008300",  # ChipGPT             (green)
    5: "#4a3aa7",  # RTLCoder            (violet)
    6: "#17becf",  # OriGen              (cyan)
    7: "#e87ba4",  # VeriCoder           (magenta)
    8: "#8b5a2b",  # VeriGen             (brown)
}
for i, c in QUAL_COLORS.items():
    colors[i] = c

p_range = np.linspace(0, 1, 500)
n = len(MODELS)
gap = 0.7  # 相邻山脊的垂直间隔

# 从下到上堆叠 (底部=CodeGen, 顶部=VeriGen)
for i in range(n - 1, -1, -1):
    d = violin_data[i]
    if len(d) < 2:
        continue
    kde = gaussian_kde(d, bw_method='scott')
    density = kde(p_range)
    density = density / density.max()  # 归一化到 [0,1]

    baseline = (n - 1 - i) * gap
    ax.fill_between(p_range, baseline, baseline + density,
                    color=colors[i], alpha=0.75, zorder=2)
    ax.plot(p_range, baseline + density, color='white', linewidth=0.6, zorder=3)

    # 模型名 (左侧, 缩短, Coder 换行居中)
    short_name = {"CodeGen-6B-Multi": "CodeGen",
                  "DeepSeek-Coder-6.7B": "DeepSeek\nCoder",
                  "Qwen2.5-Coder": "Qwen2.5\nCoder"}.get(model_names[i], model_names[i])
    ax.text(-0.12, baseline + 0.35, short_name,
            ha='center', va='center', fontsize=6.5)

    # 平均值圆点 + 数值 (颜色为山脊的加深版)
    mean_p = np.mean(d)
    mean_idx = np.argmin(np.abs(p_range - mean_p))
    mean_h = density[mean_idx]
    r, g, b = mcolors.to_rgb(colors[i])
    dark = (r * 0.55, g * 0.55, b * 0.55)
    ax.plot(mean_p, baseline + mean_h, 'o', color=dark, markersize=4,
            markeredgecolor='white', markeredgewidth=0.6, zorder=6)
    # 数值标注在圆点旁边(避开边界), 指定模型强制左/右
    if model_names[i] in ('Qwen2.5-Coder', 'OriGen'):
        ax.text(mean_p + 0.02, baseline + mean_h, f'{mean_p:.3f}',
                ha='left', va='center', fontsize=6.5, color=dark)
    elif model_names[i] == 'ChipGPT':
        ax.text(mean_p - 0.02, baseline + mean_h, f'{mean_p:.3f}',
                ha='right', va='center', fontsize=6.5, color=dark)
    elif mean_p < 0.5:
        ax.text(mean_p + 0.02, baseline + mean_h, f'{mean_p:.3f}',
                ha='left', va='center', fontsize=6.5, color=dark)
    else:
        ax.text(mean_p - 0.02, baseline + mean_h, f'{mean_p:.3f}',
                ha='right', va='center', fontsize=6.5, color=dark)

# 阈值垂直虚线 (顶部到 CodeGen 山峰高度)
top_y = (n - 1) * gap + 1.0
ax.plot([best_t, best_t], [-0.3, top_y], color='#C0392B', linestyle='--', linewidth=1.2, zorder=6)
ax.text(best_t + 0.01, top_y, f'$\\tau^*$={best_t:.3f}',
        ha='left', va='bottom', fontsize=8.5, color='#C0392B')

ax.set_xlim(0, 1.0)
ax.set_ylim(-0.3, (n - 1) * gap + 1.5)
ax.set_xlabel('Contamination probability')
ax.set_yticks([])
ax.set_ylabel('')

for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_linewidth(0.7)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")
print(f"Saved -> {OUT_PDF}")
