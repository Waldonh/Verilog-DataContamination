#!/usr/bin/env python3
"""污染集中度分析: 洛伦兹曲线 + Gini 系数 (9 model, 只画有实质污染的 model)

对每个 model, 将其在 6 个 benchmark 上的所有样本按污染概率 p 降序排列,
画累积污染质量占比曲线 (洛伦兹曲线)。
  横轴 = 累积样本比例 (最污染 -> 最干净)
  纵轴 = 累积污染质量占比 sum(p_top) / sum(p_all)
读法: 曲线越在对角线上方凸起 -> 污染越集中 (少数样本占大部分质量);
      贴近对角线 -> 污染弥漫 (质量均匀分布)。
Gini = 2*AUC - 1  (AUC 为曲线下面积), 0=均匀, 1=极端集中。
"""
import json, hashlib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# 9 个 model 的颜色, 与山脊图一一对应 (index 即 MODELS 序号)。
# CodeGen/Phi-2 保持 husl 原色(粉红/橙黄, 本图已排除不画, 仅保持索引一致);
# 其余 7 个有污染 model 用高区分度色 (避开 husl 红橙, 白底可辨)
MODEL_COLORS = [
    "#f77189",   # CodeGen-6B-Multi    (husl 粉红, 不画)
    "#d58a00",   # Phi-2               (husl 橙黄, 不画)
    "#2a78d6",   # DeepSeek-Coder-6.7B (blue)
    "#1baf7a",   # Qwen2.5-Coder       (teal)
    "#008300",   # ChipGPT             (green)
    "#4a3aa7",   # RTLCoder            (violet)
    "#17becf",   # OriGen              (cyan)
    "#e87ba4",   # VeriCoder           (magenta)
    "#8b5a2b",   # VeriGen             (brown)
]

BENCHMARKS = ["VerilogEval_v1", "VerilogEval_v2", "RTLLM_v1", "RTLLM_v2",
              "RealBench_module", "RealBench_system"]

# 图例用简短名
SHORT_NAMES = {
    "CodeGen-6B-Multi": "CodeGen",
    "DeepSeek-Coder-6.7B": "DeepSeek-Coder",
    "Qwen2.5-Coder": "Qwen2.5-Coder",
    "ChipGPT": "ChipGPT",
    "RTLCoder": "RTLCoder",
    "OriGen": "OriGen",
    "VeriCoder": "VeriCoder",
    "VeriGen": "VeriGen",
}

# 只对污染率 >= 该阈值的 model 画洛伦兹曲线 (低于此值污染质量趋于零, Gini 无定义)
MIN_RATE = 0.10

# 显式排除的近干净 model (污染质量趋于零, Gini 无定义, 不纳入)
EXCLUDE = {"CodeGen-6B-Multi", "Phi-2"}

OUT_PNG = str(BASE_DIR / "figures" / "fig_lorenz_contamination.png")
OUT_PDF = str(BASE_DIR / "figures" / "fig_lorenz_contamination.pdf")


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
print(f"OOF 阈值 tau* = {best_t:.4f} (OOF F1={best_f:.4f})")

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

# ===== 3. 洛伦兹曲线 + Gini =====
def lorenz_gini(p):
    """返回 (x, y, gini)。x=累积样本比例, y=累积污染质量占比 (p 降序)。"""
    p = np.asarray(p, dtype=float)
    p = p[~np.isnan(p)]
    if p.sum() <= 1e-12 or len(p) < 2:
        return None, None, None
    order = np.argsort(-p)          # 降序
    p_sorted = p[order]
    n = len(p)
    x = np.arange(1, n + 1) / n
    y = np.cumsum(p_sorted) / p_sorted.sum()
    auc = np.trapz(y, x)            # 曲线在对角线上方, AUC 越大越集中
    gini = 2.0 * auc - 1.0
    return x, y, gini

# 汇总每个 model 的污染率 + Gini
print("\n=== 污染率 & Gini 汇总 ===")
print(f"{'model':22s} {'n_samples':>9s} {'rate':>6s} {'Gini':>6s}  纳入?")
keep = []  # (display_name, x, y, gini, color_index)
for idx, (display_name, _) in enumerate(MODELS):
    p = probs_by_model[display_name]
    rate = float((p >= best_t).mean())
    x, y, gini = lorenz_gini(p)
    g_str = f"{gini:.3f}" if gini is not None else "  n/a"
    inc = (display_name not in EXCLUDE) and rate >= MIN_RATE and gini is not None
    flag = "Y" if inc else "  -"
    print(f"{display_name:22s} {len(p):9d} {rate:6.1%} {g_str:>6s}  {flag}")
    if inc:
        keep.append((display_name, x, y, gini, idx))

# ===== 4. 画图 =====
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

# 先算所有偏差曲线的纵轴上限 (z = y - x, 恒 >= 0)
max_z = max((y - x).max() for _, x, y, _, _ in keep)

fig, ax = plt.subplots(figsize=(3.5, 3.3))  # IEEE 单栏宽度

# 均匀基准水平线 z = 0
ax.axhline(0, color='#888888', linestyle='--', linewidth=0.9, zorder=1)

# 每条 model 一条偏差曲线 (Gini 降序: 图例集中的在前; zorder 递增: 集中的画在上层)
keep_sorted = sorted(keep, key=lambda t: -t[3])
for i, (display_name, x, y, gini, idx) in enumerate(keep_sorted):
    z = y - x
    ax.plot(x, z, color=MODEL_COLORS[idx], linewidth=1.4, zorder=3 + i,
            label=f"{SHORT_NAMES.get(display_name, display_name)} "
                  f"($\\mathrm{{Gini}}={gini:.2f}$)")

ax.set_xlim(0, 1)
ax.set_ylim(0, max_z * 1.15)
ax.set_xlabel('Cumulative fraction of samples')
ax.set_ylabel('Excess contamination mass share ($y-x$)')
ax.set_xticks(np.arange(0, 1.01, 0.2))

# 四边黑线封闭 (同柱状图风格)
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(True)
    ax.spines[s].set_linewidth(0.8)
    ax.spines[s].set_color('black')

# 图例: 右下角, 无框
ax.legend(loc='upper right', frameon=False, fontsize=6.5, handlelength=1.4,
          borderaxespad=0.5, labelspacing=0.45)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")
print(f"Saved -> {OUT_PDF}")
print(f"纳入洛伦兹图的 model 数: {len(keep)}")
