#!/usr/bin/env python3
"""集成消融: CatBoost + XGBoost + LightGBM 各种组合 (最优5f + C+O + none)"""
import json, numpy as np, pandas as pd, random
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

random.seed(42)
np.random.seed(42)

OPTIMAL_5F = ["DCPDD","Reference","neighbor","mink_0.2","zlib"]
TEST_MODELS = ["CodeV", "RTLCoder", "OriGen"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

GOLDEN_DIR = DATA_DIR / "golden"                          # golden 数据集
FEAT_DIR   = DATA_DIR / "features" / "golden"             # CodeV/OriGen 特征
OUT_FILE   = DATA_DIR / "features" / "golden" / "ablation_ensemble.json"

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
X_train = pd.concat(train_parts, ignore_index=True)[OPTIMAL_5F].astype(float)
y_train = np.concatenate(train_ys)
print(f"Train: {len(X_train)} samples ({sum(y_train)} pos)")

# 测试
test_dfs = {}; test_ys = {}
for m in TEST_MODELS:
    te = SPLITS[m]["test"]
    test_dfs[m] = all_feats[m].iloc[te][OPTIMAL_5F].astype(float)
    test_ys[m] = all_labels[m][te]

# ---- 基分类器 ----
BASE_MODELS = {
    "CatBoost": CatBoostClassifier(
        iterations=900, depth=6, learning_rate=0.03,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=42, verbose=False, allow_writing_files=False),
    "XGBoost": XGBClassifier(
        n_estimators=700, max_depth=4, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, verbosity=0),
    "LightGBM": LGBMClassifier(
        n_estimators=900, num_leaves=31, learning_rate=0.02,
        subsample=0.9, colsample_bytree=0.9,
        objective="binary", random_state=42, verbose=-1),
}

def evaluate(name, preds_fn):
    """preds_fn(model_name) -> prob array"""
    r = {}
    for tgt in TEST_MODELS:
        prob = preds_fn(tgt)
        r[tgt] = {"AUC": roc_auc_score(test_ys[tgt], prob),
                   "F1": best_f1(test_ys[tgt], prob)}
    r["Avg_AUC"] = np.mean([r[t]["AUC"] for t in TEST_MODELS])
    r["Avg_F1"]  = np.mean([r[t]["F1"] for t in TEST_MODELS])
    return r

# ---- 1. 单模型 (baseline) ----
print(f"\n{'='*80}")
print("1. Single Models")
print(f"{'='*80}")
single_results = {}
for name, clf in BASE_MODELS.items():
    clf.fit(X_train, y_train)
    single_results[name] = evaluate(name, lambda tgt, clf=clf: clf.predict_proba(test_dfs[tgt])[:,1])

# ---- 2. Simple Average ----
print("\n2. Simple Average")
avg_preds = {}
for tgt in TEST_MODELS:
    probs = []
    for name, clf in BASE_MODELS.items():
        probs.append(clf.predict_proba(test_dfs[tgt])[:,1])
    avg_preds[tgt] = np.mean(probs, axis=0)
avg_results = evaluate("Average", lambda tgt: avg_preds[tgt])

# ---- 3. Weighted Average (by single-model train AUC) ----
print("3. Weighted Average (by train OOF AUC)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_weights = {}
for name, clf_factory in [
    ("CatBoost", lambda: CatBoostClassifier(iterations=900, depth=6, learning_rate=0.03,
        loss_function="Logloss", eval_metric="AUC", random_seed=42, verbose=False, allow_writing_files=False)),
    ("XGBoost", lambda: XGBClassifier(n_estimators=700, max_depth=4, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss", random_state=42, verbosity=0)),
    ("LightGBM", lambda: LGBMClassifier(n_estimators=900, num_leaves=31, learning_rate=0.02,
        subsample=0.9, colsample_bytree=0.9, objective="binary", random_state=42, verbose=-1)),
]:
    aucs = []
    for tr_idx, val_idx in skf.split(X_train, y_train):
        clf = clf_factory()
        clf.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        prob = clf.predict_proba(X_train.iloc[val_idx])[:,1]
        aucs.append(roc_auc_score(y_train[val_idx], prob))
    oof_weights[name] = np.mean(aucs)
    print(f"  {name} OOF AUC: {oof_weights[name]:.4f}")

total_w = sum(oof_weights.values())
weighted_preds = {}
for tgt in TEST_MODELS:
    wp = np.zeros(len(test_dfs[tgt]))
    for name, clf in BASE_MODELS.items():
        wp += (oof_weights[name] / total_w) * clf.predict_proba(test_dfs[tgt])[:,1]
    weighted_preds[tgt] = wp
weighted_results = evaluate("WeightedAvg", lambda tgt: weighted_preds[tgt])

# ---- 4. Stacking (5-fold OOF → LR meta-learner) ----
print("\n4. Stacking (OOF → LogisticRegression)")
meta_X = np.zeros((len(X_train), len(BASE_MODELS)))
for i, (name, clf_factory) in enumerate([
    ("CatBoost", lambda: CatBoostClassifier(iterations=900, depth=6, learning_rate=0.03,
        loss_function="Logloss", eval_metric="AUC", random_seed=42, verbose=False, allow_writing_files=False)),
    ("XGBoost", lambda: XGBClassifier(n_estimators=700, max_depth=4, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss", random_state=42, verbosity=0)),
    ("LightGBM", lambda: LGBMClassifier(n_estimators=900, num_leaves=31, learning_rate=0.02,
        subsample=0.9, colsample_bytree=0.9, objective="binary", random_state=42, verbose=-1)),
]):
    for tr_idx, val_idx in skf.split(X_train, y_train):
        clf = clf_factory()
        clf.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        meta_X[val_idx, i] = clf.predict_proba(X_train.iloc[val_idx])[:,1]

# 目标模型: 基模型在全部训练数据上 retrain → 目标测试集预测 → meta feature
meta = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42)
meta.fit(meta_X, y_train)

stacking_preds = {}
for tgt in TEST_MODELS:
    test_meta = np.zeros((len(test_dfs[tgt]), len(BASE_MODELS)))
    for i, (name, clf) in enumerate(BASE_MODELS.items()):
        test_meta[:, i] = clf.predict_proba(test_dfs[tgt])[:,1]
    stacking_preds[tgt] = meta.predict_proba(test_meta)[:,1]
stacking_results = evaluate("Stacking", lambda tgt: stacking_preds[tgt])

# ---- 5. 两两 Stacking ----
print("5. Pairwise Stacking")
pairwise = {}
for pair_name, models in [("CAT+XGB", ["CatBoost","XGBoost"]),
                           ("CAT+LGB", ["CatBoost","LightGBM"]),
                           ("XGB+LGB", ["XGBoost","LightGBM"])]:
    # 简化为平均（无需再训 stacking）
    pp = {}
    for tgt in TEST_MODELS:
        probs = [BASE_MODELS[m].predict_proba(test_dfs[tgt])[:,1] for m in models]
        pp[tgt] = np.mean(probs, axis=0)
    pairwise[pair_name] = evaluate(pair_name, lambda tgt, pp=pp: pp[tgt])

# ---- 表格输出 ----
print(f"\n{'='*90}")
print(f"{'Method':<16} {'CodeV_AUC':>10} {'CodeV_F1':>10} {'RTLC_AUC':>10} {'RTLC_F1':>10} {'OriG_AUC':>10} {'OriG_F1':>10} {'Avg_AUC':>10} {'Avg_F1':>10}")
print("-"*100)

all_r = {}
all_r.update(single_results)
all_r["Average(3)"] = avg_results
all_r["WeightedAvg"] = weighted_results
all_r["Stacking(3)"] = stacking_results
all_r.update(pairwise)

for name in ["CatBoost","XGBoost","LightGBM","Average(3)","WeightedAvg","Stacking(3)","CAT+XGB","CAT+LGB","XGB+LGB"]:
    if name not in all_r: continue
    r = all_r[name]
    print(f"{name:<16} {r['CodeV']['AUC']:>10.4f} {r['CodeV']['F1']:>10.4f} "
          f"{r['RTLCoder']['AUC']:>10.4f} {r['RTLCoder']['F1']:>10.4f} "
          f"{r['OriGen']['AUC']:>10.4f} {r['OriGen']['F1']:>10.4f} "
          f"{r['Avg_AUC']:>10.4f} {r['Avg_F1']:>10.4f}")

with open(OUT_FILE, "w") as f:
    json.dump(all_r, f, indent=2)
print(f"\nSaved → {OUT_FILE}")
