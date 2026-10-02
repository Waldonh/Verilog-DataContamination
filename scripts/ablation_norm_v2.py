#!/usr/bin/env python3
"""归一化消融: none vs zscore vs minmax (CatBoost+最优6f)"""
import json, numpy as np, pandas as pd, random
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from catboost import CatBoostClassifier

random.seed(42)

OPTIMAL_5F = ["DCPDD","Reference","neighbor","mink_0.2","zlib"]
TEST_MODELS = ["CodeV", "RTLCoder", "OriGen"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

GOLDEN_DIR = DATA_DIR / "golden"                          # golden 数据集
FEAT_DIR   = DATA_DIR / "features" / "golden"             # CodeV/OriGen 特征
OUT_FILE   = DATA_DIR / "features" / "golden" / "ablation_norm.json"

SCORER_MAP = {
    "mink_family":["zlib","mink_0.2","mink++_0.2"], "dcpdd_ppl":["DCPDD","ppl"],
    "petal":["PETAL"], "neighbor":["neighbor"], "reference":["Reference"],
    "token_overlap":["Token_Overlap"], "cdd":["CDD"],
}

def code_hash(c):
    import hashlib
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
    best_t, best = 0.5, -1.0
    for t in np.arange(0.05, 0.951, 0.005):
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
all_feats_raw = {}; all_labels = {}
for m in TEST_MODELS:
    df, lb = load_features(m)
    all_feats_raw[m] = df; all_labels[m] = lb
    print(f"  {m}: {len(df)}x{len(df.columns)}")
SPLITS = build_splits()

# 归一化模式
NORM_MODES = {
    "none":   lambda df: df.copy(),
    "zscore": lambda df: (df - df.mean()) / df.std().replace(0, 1.0),
    "minmax": lambda df: (df - df.min()) / (df.max() - df.min()).replace(0, 1.0),
}

print(f"\n{'='*80}")
print(f"{'Norm':<8} {'CodeV_AUC':>10} {'CodeV_F1':>10} {'RTLC_AUC':>10} {'RTLC_F1':>10} {'OriG_AUC':>10} {'OriG_F1':>10} {'Avg_AUC':>10} {'Avg_F1':>10}")
print("-"*90)

all_results = {}

for norm_name, norm_fn in NORM_MODES.items():
    # 逐模型归一化
    all_feats = {}
    for m in TEST_MODELS:
        df = norm_fn(all_feats_raw[m][OPTIMAL_5F].astype(float)).fillna(0.0)
        all_feats[m] = df

    # 训练
    train_parts = []; train_ys = []
    for m in ["CodeV","OriGen"]:
        tr = SPLITS[m]["train"]
        train_parts.append(all_feats[m].iloc[tr]); train_ys.append(all_labels[m][tr])
    X_train = pd.concat(train_parts, ignore_index=True)
    y_train = np.concatenate(train_ys)

    model = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.03,
                               loss_function="Logloss", eval_metric="AUC",
                               random_seed=42, verbose=False, allow_writing_files=False)
    model.fit(X_train, y_train)

    r = {}
    row = f"{norm_name:<8}"
    for tgt in TEST_MODELS:
        te = SPLITS[tgt]["test"]
        prob = model.predict_proba(all_feats[tgt].iloc[te])[:,1]
        auc = roc_auc_score(all_labels[tgt][te], prob)
        f1  = best_f1(all_labels[tgt][te], prob)
        r[tgt] = {"AUC": auc, "F1": f1}
        row += f" {auc:>10.4f} {f1:>10.4f}"
    r["Avg_AUC"] = np.mean([r[t]["AUC"] for t in TEST_MODELS])
    r["Avg_F1"]  = np.mean([r[t]["F1"] for t in TEST_MODELS])
    row += f" {r['Avg_AUC']:>10.4f} {r['Avg_F1']:>10.4f}"
    all_results[norm_name] = r
    print(row)

with open(OUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved → {OUT_FILE}")
