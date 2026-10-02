#!/usr/bin/env python3
"""CatBoost-5f vs 10单探针 baseline (最优配置: 5f + C+O + none)"""
import json, numpy as np, pandas as pd, random, hashlib
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

random.seed(42)

OPTIMAL_5F = ["DCPDD","Reference","neighbor","mink_0.2","zlib"]
ALL_PROBES = ["zlib","mink_0.2","mink++_0.2","DCPDD","ppl","PETAL","neighbor","Reference","Token_Overlap","CDD"]
TEST_MODELS = ["CodeV","RTLCoder","OriGen"]

SCORER_MAP = {
    "mink_family":["zlib","mink_0.2","mink++_0.2"], "dcpdd_ppl":["DCPDD","ppl"],
    "petal":["PETAL"], "neighbor":["neighbor"], "reference":["Reference"],
    "token_overlap":["Token_Overlap"], "cdd":["CDD"],
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

GOLDEN_DIR = DATA_DIR / "golden"                          # golden 数据集
FEAT_DIR   = DATA_DIR / "features" / "golden"             # CodeV/OriGen 特征

def code_hash(c):
    return hashlib.sha256(c.encode()).hexdigest()

def load_features(model):
    feat_dir = FEAT_DIR / model
    combined = {}; labels = None
    for subd, cols in SCORER_MAP.items():
        path = feat_dir / subd / "results.jsonl"
        if not path.exists(): continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if labels is None: labels = np.array([int(r.get("label",0)) for r in rows])
        for col in cols:
            actual = {"Token_Overlap":"contamination_score","CDD":"contamination_score",
                      "Reference":"ref","ppl":"PPL"}.get(col, col)
            combined[col] = [float(r.get(actual, np.nan)) for r in rows[:len(labels)]]
    if not combined: return None, None
    min_len = min(len(v) for v in combined.values())
    return pd.DataFrame({k:v[:min_len] for k,v in combined.items()}), labels[:min_len]

def best_f1(y_true, prob):
    """阈值搜索最大化 F1: 对概率/归一化分数搜 [0.05,0.95], 对原始分数搜实际范围"""
    # 自动检测: 如果值在 [0,1] 之间, 用标准搜索; 否则搜分数分布的分位数范围
    if prob.min() >= 0 and prob.max() <= 1:
        rng = np.arange(0.05, 0.951, 0.005)
    else:
        # 搜实际分数范围的 5%-95% 分位数
        lo, hi = np.percentile(prob, [1, 99])
        rng = np.linspace(lo, hi, 200)
    best_t, best = 0.5, -1.0
    for t in rng:
        f1 = f1_score(y_true, (prob >= t).astype(int))
        if f1 > best: best_t, best = float(t), f1
    return f1_score(y_true, (prob >= best_t).astype(int))

def build_splits():
    splits = {}
    for m in TEST_MODELS:
        all_path = FEAT_DIR / f"{m}_all.jsonl"
        all_rows = [json.loads(l) for l in open(all_path, encoding="utf-8") if l.strip()]
        all_hashes = [code_hash(r["Response"]) for r in all_rows]
        test_path = GOLDEN_DIR / f"{m}_test_200.jsonl"
        test_hashes = set()
        for line in open(test_path, encoding="utf-8"):
            if line.strip(): test_hashes.add(code_hash(json.loads(line)["Response"]))
        te_idx = [i for i,h in enumerate(all_hashes) if h in test_hashes]
        tr_idx = [i for i in range(len(all_hashes)) if i not in set(te_idx)]
        splits[m] = {"train": tr_idx, "test": te_idx}
    return splits

# ---- 加载 ----
print("Loading...")
all_feats = {}; all_labels = {}
for m in TEST_MODELS:
    df, lb = load_features(m)
    all_feats[m] = df; all_labels[m] = lb
    print(f"  {m}: {len(df)}x{len(df.columns)}")
SPLITS = build_splits()

# 训练: CodeV + OriGen
train_parts = []; train_ys = []
for m in ["CodeV","OriGen"]:
    tr = SPLITS[m]["train"]
    train_parts.append(all_feats[m].iloc[tr]); train_ys.append(all_labels[m][tr])
train_df = pd.concat(train_parts, ignore_index=True)
train_y = np.concatenate(train_ys)
print(f"Train: {len(train_df)} samples (C+O)")

# ---- CatBoost-5f ----
model = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.03,
                           loss_function="Logloss", eval_metric="AUC",
                           random_seed=42, verbose=False, allow_writing_files=False)
model.fit(train_df[OPTIMAL_5F].astype(float), train_y)

# OOF 阈值 τ* (5 折分层 CV, 最大化 OOF F1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train_y))
for tr_idx, val_idx in skf.split(train_df[OPTIMAL_5F].astype(float), train_y):
    m = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.03,
                           loss_function="Logloss", eval_metric="AUC",
                           random_seed=42, verbose=False, allow_writing_files=False)
    m.fit(train_df[OPTIMAL_5F].iloc[tr_idx].astype(float), train_y[tr_idx])
    oof[val_idx] = m.predict_proba(train_df[OPTIMAL_5F].iloc[val_idx].astype(float))[:, 1]
best_t, best_f = 0.5, -1.0
for t in np.arange(0.05, 0.951, 0.005):
    f = f1_score(train_y, (oof >= t).astype(int))
    if f > best_f: best_t, best_f = float(t), f
print(f"OOF 阈值 τ* = {best_t:.4f} (OOF F1={best_f:.4f})")

catboost_results = {}
for tgt in TEST_MODELS:
    te = SPLITS[tgt]["test"]
    prob = model.predict_proba(all_feats[tgt].iloc[te][OPTIMAL_5F].astype(float))[:,1]
    yt = all_labels[tgt][te]
    catboost_results[tgt] = {"AUC": roc_auc_score(yt, prob),
                             "F1": f1_score(yt, (prob >= best_t).astype(int))}

# ---- 单探针 baseline ----
print("Computing single-probe baselines...")
single_results = {}
for col in ALL_PROBES:
    single_results[col] = {}
    for tgt in TEST_MODELS:
        te = SPLITS[tgt]["test"]
        vals = all_feats[tgt].iloc[te][col].values.astype(float)
        yt = all_labels[tgt][te]
        auc = roc_auc_score(yt, vals)
        if auc < 0.5: auc = roc_auc_score(yt, -vals)
        single_results[col][tgt] = {"AUC": auc, "F1": best_f1(yt, vals)}

# ---- 表格 ----
print(f"\n{'='*100}")
print(f"{'Method':<16} {'CodeV_AUC':>10} {'CodeV_F1':>10} {'RTLC_AUC':>10} {'RTLC_F1':>10} {'OriG_AUC':>10} {'OriG_F1':>10} {'Avg_AUC':>10} {'Avg_F1':>10}")
print("-"*100)

for col in ALL_PROBES:
    row = f"{col:<16}"; aucs = []; f1s = []
    for tgt in TEST_MODELS:
        r = single_results[col][tgt]
        row += f" {r['AUC']:>10.4f} {r['F1']:>10.4f}"
        aucs.append(r['AUC']); f1s.append(r['F1'])
    row += f" {np.mean(aucs):>10.4f} {np.mean(f1s):>10.4f}"
    print(row)

print("-"*100)
row = f"{'CatBoost-5f':<16}"; aucs = []; f1s = []
for tgt in TEST_MODELS:
    r = catboost_results[tgt]
    row += f" {r['AUC']:>10.4f} {r['F1']:>10.4f}"
    aucs.append(r['AUC']); f1s.append(r['F1'])
row += f" {np.mean(aucs):>10.4f} {np.mean(f1s):>10.4f}"
print(row)
print("-"*100)

# 提幅
best_single_auc = max(np.mean([single_results[col][t]["AUC"] for t in TEST_MODELS]) for col in ALL_PROBES)
print(f"\nCatBoost-5f Avg AUC: {np.mean(aucs):.4f}")
print(f"Best single probe Avg AUC: {best_single_auc:.4f}")
print(f"Improvement: {np.mean(aucs) - best_single_auc:+.4f}")

print("\n[DONE]")
