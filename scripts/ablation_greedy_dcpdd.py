#!/usr/bin/env python3
"""贪心前向选择消融: 从 DCPDD 开始, 每次加一个使 Avg AUC 最高的探针"""
import json, numpy as np, pandas as pd, random
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from catboost import CatBoostClassifier

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

GOLDEN_DIR = DATA_DIR / "golden"                          # golden 数据集
FEAT_DIR   = DATA_DIR / "features" / "golden"             # CodeV/OriGen 特征
OUT_FILE   = DATA_DIR / "features" / "golden" / "greedy_dcpdd.json"

ALL_PROBES = ["zlib","mink_0.2","mink++_0.2","DCPDD","ppl","PETAL","neighbor","Reference","Token_Overlap","CDD"]
TEST_MODELS = ["CodeV", "RTLCoder", "OriGen"]

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
train_df = pd.concat(train_parts, ignore_index=True)
train_y = np.concatenate(train_ys)

# 测试 diffs
test_dfs = {}; test_ys = {}
for m in TEST_MODELS:
    te = SPLITS[m]["test"]
    test_dfs[m] = all_feats[m].iloc[te]; test_ys[m] = all_labels[m][te]

def evaluate(feats):
    """给定特征列表, 训练 CatBoost, 返回各模型 AUC/F1 + Avg AUC"""
    m = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.03,
                           loss_function="Logloss", eval_metric="AUC",
                           random_seed=42, verbose=False, allow_writing_files=False)
    m.fit(train_df[feats].astype(float), train_y)
    results = {}
    for tgt in TEST_MODELS:
        prob = m.predict_proba(test_dfs[tgt][feats].astype(float))[:,1]
        results[tgt] = {"AUC": roc_auc_score(test_ys[tgt], prob),
                         "F1": best_f1(test_ys[tgt], prob)}
    results["Avg_AUC"] = np.mean([results[t]["AUC"] for t in TEST_MODELS])
    return results

# ---- 贪心前向 ----
start = "DCPDD"
selected = [start]
candidates = [p for p in ALL_PROBES if p != start]
results_log = []

print(f"\n{'='*70}")
print(f"Greedy Forward from '{start}'")
print(f"{'='*70}")

r = evaluate(selected)
results_log.append({"features": selected[:], "results": r})
print(f"\nStart [{','.join(selected)}]: Avg AUC = {r['Avg_AUC']:.4f}")

for step in range(1, len(ALL_PROBES)):
    best_probe = None
    best_avg = -1.0
    best_detail = None

    for cand in candidates:
        cur_feats = selected + [cand]
        r = evaluate(cur_feats)
        if r["Avg_AUC"] > best_avg:
            best_avg = r["Avg_AUC"]
            best_probe = cand
            best_detail = r

    if best_probe is None:
        break

    selected.append(best_probe)
    candidates.remove(best_probe)
    results_log.append({"features": selected[:], "results": best_detail})

    detail = " ".join([f"{t[:4]}={best_detail[t]['AUC']:.4f}" for t in TEST_MODELS])
    print(f"  + {best_probe:<14} → Avg AUC = {best_avg:.4f}  [{detail}]")

# ---- 输出表格 ----
print(f"\n{'='*70}")
print(f"{'K':<3} {'Features':<60} {'Avg_AUC':>8}  {'C_AUC':>6} {'R_AUC':>6} {'O_AUC':>6}")
print("-"*95)
for entry in results_log:
    feats = entry["features"]
    r = entry["results"]
    line = ",".join(feats)
    print(f"{len(feats):<3} {line:<60} {r['Avg_AUC']:>8.4f}  {r['CodeV']['AUC']:>6.4f} {r['RTLCoder']['AUC']:>6.4f} {r['OriGen']['AUC']:>6.4f}")

# 保存
with open(OUT_FILE, "w") as f:
    json.dump(results_log, f, indent=2)
print(f"\nSaved → {OUT_FILE}")
