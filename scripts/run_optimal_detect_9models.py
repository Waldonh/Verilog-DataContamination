#!/usr/bin/env python3
"""
最优配置探测器 (CatBoost-5f, C+O) 检测 9 个模型 x 6 个 benchmark 的污染率。

- 训练: CodeV + OriGen, 5f, CatBoost, none, OOF 阈值
- 检测: 9 模型 x 6 benchmark, 输出污染率表 + results.json
- 特征: benchmark_detect_12/features (5f, 已更新)

用法(服务器):
  python run_optimal_detect_9models.py
"""
import json, hashlib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

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

# 9 个模型 (特征目录名)
TARGET_MODELS = [
    "CodeGen-6B-Multi", "Phi-2", "DeepSeek-Coder-6.7B", "Qwen2.5-Coder-7B",
    "ChipGPT-Llama2-7B", "RTLCoder", "OriGen", "VeriCoder", "FineTuned-CodeGen-6B",
]

BENCHMARKS = ["VerilogEval_v1", "VerilogEval_v2", "RTLLM_v1", "RTLLM_v2",
              "RealBench_module", "RealBench_system"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

GOLDEN_DIR = DATA_DIR / "golden"                          # golden 数据集
GOLDEN_FEAT_DIR = DATA_DIR / "features" / "golden"         # CodeV/OriGen 特征
BENCH_FEAT_DIR = DATA_DIR / "features" / "benchmark"       # benchmark 特征
OUT_FILE = DATA_DIR / "features" / "benchmark" / "results_9models.json"


def code_hash(c):
    return hashlib.sha256(c.encode("utf-8")).hexdigest()


def load_features(feat_dir):
    feat_dir = Path(feat_dir); combined = {}; labels = None
    for subd, cols in SCORER_MAP.items():
        path = feat_dir / subd / "results.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
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
    df, lb = load_features(GOLDEN_FEAT_DIR / m)
    train_feats[m] = df; train_labels[m] = lb

splits = {}
for m in ["CodeV", "OriGen"]:
    all_path = GOLDEN_FEAT_DIR / f"{m}_all.jsonl"
    all_rows = [json.loads(l) for l in open(all_path, encoding="utf-8") if l.strip()]
    all_hashes = [code_hash(r["Response"]) for r in all_rows]
    test_path = GOLDEN_DIR / f"{m}_test_200.jsonl"
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
best_t, best_f = 0.5, -1.0
for t in np.arange(0.05, 0.951, 0.005):
    f = f1_score(y_train, (oof >= t).astype(int))
    if f > best_f:
        best_t, best_f = float(t), f
print(f"OOF 阈值 τ* = {best_t:.4f} (OOF F1={best_f:.4f})\n")

# ===== 2. 检测 9 模型 x 6 benchmark =====
results = {}
for mname in TARGET_MODELS:
    for bname in BENCHMARKS:
        feat_dir = BENCH_FEAT_DIR / mname / bname
        df, _ = load_features(feat_dir)
        if df is None or len(df) == 0:
            print(f"  SKIP {mname}/{bname}: no features")
            continue
        X = df[OPTIMAL_5F].astype(float).fillna(0.0)
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= best_t).astype(int)
        n_contam = int(pred.sum())
        rate = n_contam / len(pred)
        results[f"{mname}|{bname}"] = {
            "model": mname, "benchmark": bname,
            "total": len(pred), "contaminated": n_contam,
            "rate": round(rate, 4), "mean_prob": round(float(prob.mean()), 4),
        }
        print(f"  {mname:<20s} {bname:<20s}: {n_contam}/{len(pred)} = {rate:.1%}  (mean p={prob.mean():.4f})")

# ===== 3. 表格输出 =====
print(f"\n{'='*90}")
header = f"{'Model':<22}"
for bname in BENCHMARKS:
    header += f" {bname[:11]:>12}"
print(header)
print("-"*90)
for mname in TARGET_MODELS:
    row = f"{mname:<22}"
    for bname in BENCHMARKS:
        r = results.get(f"{mname}|{bname}")
        row += f" {r['rate']:>11.1%}" if r else f" {'N/A':>12}"
    print(row)

# ===== 4. 统计 =====
# 每个 benchmark 的平均污染率 (跨 9 model, 总污染样本数/总样本数)
bench_avg = {}
for bname in BENCHMARKS:
    contams = [results[f"{m}|{bname}"]["contaminated"] for m in TARGET_MODELS if f"{m}|{bname}" in results]
    totals = [results[f"{m}|{bname}"]["total"] for m in TARGET_MODELS if f"{m}|{bname}" in results]
    if totals:
        bench_avg[bname] = sum(contams) / sum(totals)

# 每个 model 的加权平均污染率 (跨 6 benchmark, 样本数加权 = 总污染样本数/总样本数)
model_weighted = {}
for mname in TARGET_MODELS:
    contams = [results[f"{mname}|{b}"]["contaminated"] for b in BENCHMARKS if f"{mname}|{b}" in results]
    totals = [results[f"{mname}|{b}"]["total"] for b in BENCHMARKS if f"{mname}|{b}" in results]
    if totals:
        model_weighted[mname] = sum(contams) / sum(totals)

print(f"\n{'='*90}")
print("每个 benchmark 的平均污染率 (跨 9 model):")
print(f"{'Benchmark':<22} {'Avg_rate':>10} {'Total_contam':>14} {'Total_samples':>14}")
for bname in BENCHMARKS:
    if bname in bench_avg:
        c = sum(results[f"{m}|{bname}"]["contaminated"] for m in TARGET_MODELS if f"{m}|{bname}" in results)
        t = sum(results[f"{m}|{bname}"]["total"] for m in TARGET_MODELS if f"{m}|{bname}" in results)
        print(f"{bname:<22} {bench_avg[bname]:>10.1%} {c:>14} {t:>14}")

print(f"\n每个 model 的加权平均污染率 (跨 6 benchmark, 样本数加权):")
print(f"{'Model':<22} {'Weighted_rate':>14} {'Total_contam':>14} {'Total_samples':>14}")
for mname in TARGET_MODELS:
    if mname in model_weighted:
        c = sum(results[f"{mname}|{b}"]["contaminated"] for b in BENCHMARKS if f"{mname}|{b}" in results)
        t = sum(results[f"{mname}|{b}"]["total"] for b in BENCHMARKS if f"{mname}|{b}" in results)
        print(f"{mname:<22} {model_weighted[mname]:>14.1%} {c:>14} {t:>14}")

# 总体加权污染率 (54 个 model-benchmark 对)
total_contam_all = sum(r["contaminated"] for r in results.values())
total_samples_all = sum(r["total"] for r in results.values())
overall_rate = total_contam_all / total_samples_all if total_samples_all else float('nan')
print(f"\n总体加权污染率 (54 个 model-benchmark 对):")
print(f"  {overall_rate:.1%}  ({total_contam_all}/{total_samples_all})")

out = {
    "results": results,
    "benchmark_avg_rate": bench_avg,
    "model_weighted_rate": model_weighted,
    "overall_weighted_rate": overall_rate,
}
with open(OUT_FILE, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved -> {OUT_FILE}")
