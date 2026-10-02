#!/usr/bin/env python3
"""训练数据量消融: 单源/双源, tc 50→400 (CatBoost+最优6f+none)"""
import json, numpy as np, pandas as pd, random
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from catboost import CatBoostClassifier

random.seed(42)

OPTIMAL_5F = ["DCPDD","Reference","neighbor","mink_0.2","zlib"]
TEST_MODELS = ["CodeV", "RTLCoder", "OriGen"]
TC_VALUES = [50, 100, 150, 200, 250, 300, 350, 400]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

GOLDEN_DIR = DATA_DIR / "golden"                          # golden 数据集
FEAT_DIR   = DATA_DIR / "features" / "golden"             # CodeV/OriGen 特征
OUT_FILE   = DATA_DIR / "features" / "golden" / "ablation_train_size.json"

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
    all_feats[m] = df[OPTIMAL_5F].astype(float)
    all_labels[m] = lb
    print(f"  {m}: {len(df)}x{len(df.columns)}")
SPLITS = build_splits()

# 每个模型的训练集按正负分离 (hash排序保证确定性)
model_train = {}
for m in ["CodeV","RTLCoder","OriGen"]:
    tr_idx = SPLITS[m]["train"]
    pos_idx = [i for i in tr_idx if all_labels[m][i] == 1]
    neg_idx = [i for i in tr_idx if all_labels[m][i] == 0]
    # 按 hash 排序取前 tc 个, 保证一致性
    all_path = FEAT_DIR / f"{m}_all.jsonl"
    all_rows = [json.loads(l) for l in open(all_path, encoding="utf-8") if l.strip()]
    pos_sorted = sorted(pos_idx, key=lambda i: code_hash(all_rows[i]["Response"]))
    neg_sorted = sorted(neg_idx, key=lambda i: code_hash(all_rows[i]["Response"]))
    model_train[m] = {"pos": pos_sorted, "neg": neg_sorted}

# 测试集
test_dfs = {}; test_ys = {}
for m in TEST_MODELS:
    te = SPLITS[m]["test"]
    test_dfs[m] = all_feats[m].iloc[te]
    test_ys[m] = all_labels[m][te]

# ---- 消融 ----
COMBOS = [
    ("C→All",        ["CodeV"]),
    ("R→All",        ["RTLCoder"]),
    ("O→All",        ["OriGen"]),
    ("C+R→All",      ["CodeV","RTLCoder"]),
    ("C+O→All",      ["CodeV","OriGen"]),
    ("R+O→All",      ["RTLCoder","OriGen"]),
    ("C+R+O→All",    ["CodeV","RTLCoder","OriGen"]),
]

print(f"\n{'='*100}")
print(f"{'Train':<10} {'tc':>4} {'CodeV_AUC':>10} {'CodeV_F1':>10} {'RTLC_AUC':>10} {'RTLC_F1':>10} {'OriG_AUC':>10} {'OriG_F1':>10} {'Avg_AUC':>10} {'Avg_F1':>10}")
print("-"*110)

all_results = {}

for label, src_models in COMBOS:
    for tc in TC_VALUES:
        # 构建训练集: 每个源模型取 tc 正 + tc 负
        train_parts = []; train_ys = []
        for m in src_models:
            p_idx = model_train[m]["pos"][:tc]
            n_idx = model_train[m]["neg"][:tc]
            idxs = p_idx + n_idx
            train_parts.append(all_feats[m].iloc[idxs])
            train_ys.append(all_labels[m][idxs])

        X_train = pd.concat(train_parts, ignore_index=True)
        y_train = np.concatenate(train_ys)

        model = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.03,
                                   loss_function="Logloss", eval_metric="AUC",
                                   random_seed=42, verbose=False, allow_writing_files=False)
        model.fit(X_train, y_train)

        r = {}
        row = f"{label:<10} {tc:>4}"
        for tgt in TEST_MODELS:
            prob = model.predict_proba(test_dfs[tgt])[:,1]
            auc = roc_auc_score(test_ys[tgt], prob)
            f1  = best_f1(test_ys[tgt], prob)
            r[tgt] = {"AUC": auc, "F1": f1}
            row += f" {auc:>10.4f} {f1:>10.4f}"
        r["Avg_AUC"] = np.mean([r[t]["AUC"] for t in TEST_MODELS])
        r["Avg_F1"]  = np.mean([r[t]["F1"] for t in TEST_MODELS])
        row += f" {r['Avg_AUC']:>10.4f} {r['Avg_F1']:>10.4f}"
        all_results[f"{label}_tc{tc}"] = r
        print(row)

with open(OUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved → {OUT_FILE}")
