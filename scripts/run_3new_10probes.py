#!/usr/bin/env python3
"""
对 HAVEN/HAVEN_CodeQwen/vericoder (新负样本) 用 10 个方法 + CatBoost-5f 性能测试
- Phase 1: 算 10 方法特征
- Phase 2: 10 单探针 AUC/F1 + CatBoost-5f AUC/F1
"""
import json, os, sys, subprocess, argparse, hashlib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score

# 10 方法: (子目录, score字段)
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

# 模型权重（自行下载到 models/）与第三方方法脚本（另装到 scripts/10_methods/）
MODEL_DIR = BASE_DIR / "models"
THIRD_PARTY = BASE_DIR / "scripts" / "10_methods"

MODELS = {
    "HAVEN":         str(MODEL_DIR / "HAVEN"),
    "HAVEN_CodeQwen": str(MODEL_DIR / "HAVEN_CodeQwen"),
    "vericoder":      str(MODEL_DIR / "vericoder"),
}
# 三个模型都用 transformers (保守)
MODEL_BACKENDS = {"HAVEN": "transformers", "HAVEN_CodeQwen": "transformers", "vericoder": "transformers"}

GOLDEN_DIR = DATA_DIR / "golden_haven"                    # HAVEN 批 golden 数据集
FEAT_ROOT = DATA_DIR / "features" / "haven"               # HAVEN 批特征
FEAT_ROOT.mkdir(parents=True, exist_ok=True)

# 最优 5f (CatBoost 用)
OPTIMAL_5F = ["DCPDD", "Reference", "neighbor", "mink_0.2", "zlib"]
# 训练数据 (C+O)
TRAIN_FEAT_DIR = DATA_DIR / "features" / "golden"         # CodeV/OriGen 特征
TRAIN_GOLDEN_DIR = DATA_DIR / "golden"                     # golden 数据集

# 参考模型 & 第三方方法脚本路径
REF_MODEL = str(MODEL_DIR / "Qwen2.5-1.5B-Instruct")
NEIGHBOR_MLM = str(MODEL_DIR / "codebert-base")
PETAL_EMB = str(MODEL_DIR / "all-MiniLM-L6-v2")
DCPDD_CORPUS = str(THIRD_PARTY / "DCPDD" / "ref_code" / "ref_verilog.jsonl")
MINK_SCRIPT = str(THIRD_PARTY / "Mink++" / "run_zlib_mink_mink++.py")
DCPDD_SCRIPT = str(THIRD_PARTY / "DCPDD" / "run_local_dcpdd_ppl_eval.py")
PETAL_SCRIPT = str(THIRD_PARTY / "PETAL" / "run.py")
NEIGHBOR_SCRIPT = str(THIRD_PARTY / "Mink++" / "run_neighbor.py")
REF_SCRIPT = str(THIRD_PARTY / "Mink++" / "run_ref.py")
TOKEN_SCRIPT = str(THIRD_PARTY / "data-contamination-main" / "code" / "Token_Overlap" / "run_local_code_overlap_eval.py")
CDD_SCRIPT = str(THIRD_PARTY / "CDD" / "run_local_cdd_eval.py")


def code_hash(c):
    return hashlib.sha256(c.encode("utf-8")).hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--py", default=sys.executable)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="0")
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--skip_features", action="store_true")
    return p.parse_args()


def run_cmd(cmd, env, expected, reuse):
    if reuse and Path(expected).exists():
        print(f"  [SKIP] {Path(expected).name}")
        return
    subprocess.run(cmd, check=True, env=env)


def load_features(feat_dir):
    feat_dir = Path(feat_dir); combined = {}; labels = None
    for subd, cols in SCORER_MAP.items():
        path = feat_dir / subd / "results.jsonl"
        if not path.exists(): continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if labels is None: labels = np.array([int(r.get("label", 0)) for r in rows])
        for col in cols:
            actual = {"Token_Overlap": "contamination_score", "CDD": "contamination_score",
                      "Reference": "ref", "ppl": "PPL"}.get(col, col)
            combined[col] = [float(r.get(actual, np.nan)) for r in rows[:len(labels)]]
    if not combined: return None, None
    min_len = min(len(v) for v in combined.values())
    return pd.DataFrame({k: v[:min_len] for k, v in combined.items()}), labels[:min_len]


def best_f1(y_true, prob):
    if prob.min() >= 0 and prob.max() <= 1:
        rng = np.arange(0.05, 0.951, 0.005)
    else:
        lo, hi = np.percentile(prob, [1, 99])
        rng = np.linspace(lo, hi, 200)
    best_t, best = 0.5, -1.0
    for t in rng:
        f1 = f1_score(y_true, (prob >= t).astype(int))
        if f1 > best: best_t, best = float(t), f1
    return f1_score(y_true, (prob >= best_t).astype(int))


# ===== Phase 1: 算 10 方法特征 =====
def compute_features(args):
    env = os.environ.copy()
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"; env["TRANSFORMERS_OFFLINE"] = "1"

    for m, mpath in MODELS.items():
        inp = GOLDEN_DIR / f"{m}_all_1000.jsonl"
        inp_prep = FEAT_ROOT / f"{m}_all.jsonl"
        # 准备输入 (加 FullText + model 字段)
        if not inp_prep.exists():
            rows = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]
            for r in rows:
                r["FullText"] = r["Instruction"] + "\n" + r["Response"]
                r["model"] = "RTLCoder"
            with open(inp_prep, "w", encoding="utf-8") as f:
                for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

        od = FEAT_ROOT / m
        fast_env = env.copy(); fast_env["CUDA_VISIBLE_DEVICES"] = args.device
        backend = MODEL_BACKENDS[m]

        # mink family (zlib, mink_0.2, mink++_0.2)
        d = od / "mink_family"; d.mkdir(parents=True, exist_ok=True)
        run_cmd([args.py, MINK_SCRIPT, "--input_jsonl", str(inp_prep),
                 "--output_jsonl", str(d/"results.jsonl"), "--summary_json", str(d/"summary.json"),
                 "--response_field", "Response", "--label_field", "label", "--model_field", "model",
                 "--rtlcoder_model_path", mpath, "--codev_model_path", mpath,
                 "--device", "cuda", "--dtype", args.dtype,
                 "--threshold_zlib", "-0.003", "--threshold_mink_0_2", "-3.0", "--threshold_minkpp_0_2", "-3.0"],
                fast_env, d/"summary.json", args.reuse)

        # dcpdd + ppl
        d = od / "dcpdd_ppl"; d.mkdir(parents=True, exist_ok=True)
        run_cmd([args.py, DCPDD_SCRIPT, "--input_jsonl", str(inp_prep),
                 "--output_jsonl", str(d/"results.jsonl"), "--summary_json", str(d/"summary.json"),
                 "--response_field", "Response", "--label_field", "label", "--model_field", "model",
                 "--rtlcoder_model_path", mpath, "--codev_model_path", mpath,
                 "--ref_model_path", REF_MODEL, "--ref_corpus_jsonl", DCPDD_CORPUS,
                 "--freq_cache_dir", str(d/"freq_cache"),
                 "--device", "cuda", "--dtype", args.dtype,
                 "--threshold_dcpdd", "-3.0", "--threshold_ppl", "-2.2"],
                fast_env, d/"summary.json", args.reuse)

        # petal
        d = od / "petal"; d.mkdir(parents=True, exist_ok=True)
        run_cmd([args.py, PETAL_SCRIPT, "--input_jsonl", str(inp_prep),
                 "--output_jsonl", str(d/"results.jsonl"), "--summary_json", str(d/"summary.json"),
                 "--response_field", "Response", "--label_field", "label", "--model_field", "model",
                 "--rtlcoder_model_path", mpath, "--codev_model_path", mpath,
                 "--surrogate_model_path", REF_MODEL, "--embedding_model_path", PETAL_EMB,
                 "--device", "cuda", "--surrogate_device", "cuda", "--embedding_device", "auto",
                 "--dtype", args.dtype, "--threshold_petal", "0.8", "--decision_rule", "lt"],
                fast_env, d/"summary.json", args.reuse)

        # neighbor
        d = od / "neighbor"; d.mkdir(parents=True, exist_ok=True)
        run_cmd([args.py, NEIGHBOR_SCRIPT, "--input_jsonl", str(inp_prep),
                 "--output_jsonl", str(d/"results.jsonl"), "--summary_json", str(d/"summary.json"),
                 "--neighbor_cache_jsonl", str(d/"neighbor_cache.jsonl"),
                 "--response_field", "Response", "--label_field", "label", "--model_field", "model",
                 "--rtlcoder_model_path", mpath, "--codev_model_path", mpath,
                 "--neighbor_mlm_model_path", NEIGHBOR_MLM,
                 "--device", "cuda", "--perturb_device", "cuda",
                 "--dtype", args.dtype, "--threshold_neighbor", "0.8"],
                fast_env, d/"summary.json", args.reuse)

        # reference
        d = od / "reference"; d.mkdir(parents=True, exist_ok=True)
        run_cmd([args.py, REF_SCRIPT, "--input_jsonl", str(inp_prep),
                 "--output_jsonl", str(d/"results.jsonl"), "--summary_json", str(d/"summary.json"),
                 "--response_field", "Response", "--label_field", "label", "--model_field", "model",
                 "--rtlcoder_model_path", mpath, "--codev_model_path", mpath,
                 "--ref_model_path", REF_MODEL,
                 "--device", "cuda", "--ref_device", "cuda",
                 "--dtype", args.dtype, "--threshold_ref", "0.13"],
                fast_env, d/"summary.json", args.reuse)

        # token overlap
        d = od / "token_overlap"; d.mkdir(parents=True, exist_ok=True)
        run_cmd([args.py, TOKEN_SCRIPT, "--input_jsonl", str(inp_prep),
                 "--output_jsonl", str(d/"results.jsonl"), "--summary_json", str(d/"summary.json"),
                 "--stats_dir", str(d/"stats"), "--generation_ckpt_jsonl", str(d/"generation_ckpt.jsonl"),
                 "--text_field", "Response", "--label_field", "label", "--model_field", "model",
                 "--rtlcoder_model_path", mpath, "--codev_model_path", mpath,
                 "--device", "cuda", "--dtype", args.dtype,
                 "--min_p", "20.0", "--max_p", "20.0", "--max_new_tokens", "128",
                 "--icl_max_input_tokens", "1536", "--score_threshold", "0.55", "--disable_bleurt_eval"],
                fast_env, d/"summary.json", args.reuse)

        # cdd (transformers backend, 小 batch)
        d = od / "cdd"; d.mkdir(parents=True, exist_ok=True)
        cdd_env = env.copy(); cdd_env["CUDA_VISIBLE_DEVICES"] = args.device
        cdd_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        run_cmd([args.py, CDD_SCRIPT, "--input_jsonl", str(inp_prep),
                 "--output_jsonl", str(d/"results.jsonl"), "--summary_json", str(d/"summary.json"),
                 "--generation_ckpt_jsonl", str(d/"generation_ckpt.jsonl"),
                 "--instruction_field", "Instruction", "--response_field", "Response",
                 "--label_field", "label", "--model_field", "model",
                 "--rtlcoder_model_path", mpath, "--codev_model_path", mpath,
                 "--backend", backend, "--device", "cuda", "--dtype", args.dtype,
                 "--gen_batch_size", "1", "--tensor_parallel_size", "1",
                 "--generation_ckpt_every", "10", "--alpha", "0.05", "--xi", "0.01",
                 "--num_samples", "50", "--max_input_tokens", "1024", "--max_new_tokens", "128",
                 "--sample_temperature", "0.8", "--sample_top_p", "1.0", "--max_compare_tokens", "100"],
                cdd_env, d/"summary.json", args.reuse)

        print(f"  [DONE] {m}")


# ===== Phase 2: 性能测试 =====
def evaluate():
    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedKFold

    # 10 个方法的显示名
    METHODS = ["zlib", "mink_0.2", "mink++_0.2", "DCPDD", "ppl", "PETAL", "neighbor", "Reference", "Token_Overlap", "CDD"]

    # 1. 训练 CatBoost-5f (C+O)
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

    # OOF 阈值
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
    print(f"CatBoost OOF 阈值 τ*={best_t:.4f} (OOF F1={best_f:.4f})")

    # 2. 三个模型测试
    METHODS_ALL = METHODS + ["CatBoost-5f"]
    agg = {name: {"auc": [], "f1": []} for name in METHODS_ALL}

    for m in MODELS:
        feat_dir = FEAT_ROOT / m
        df, labels = load_features(feat_dir)
        if df is None:
            print(f"{m}: 特征未找到")
            continue

        all_path = FEAT_ROOT / f"{m}_all.jsonl"  # 输入文件在 features 目录下, 不在 model 子目录
        all_rows = [json.loads(l) for l in open(all_path, encoding="utf-8") if l.strip()]
        all_hashes = [code_hash(r["Response"]) for r in all_rows]
        test_path = GOLDEN_DIR / f"{m}_test_200.jsonl"
        test_hashes = {code_hash(json.loads(l)["Response"]) for l in open(test_path, encoding="utf-8") if l.strip()}
        te_idx = [i for i, h in enumerate(all_hashes) if h in test_hashes]
        y_test = labels[te_idx]

        print(f"\n{'='*60}")
        print(f"  {m}  (test={len(te_idx)}, pos={int(y_test.sum())})")
        print(f"{'='*60}")
        print(f"{'Method':<16} {'AUC':>8} {'F1':>8}")

        for col in METHODS:
            if col not in df.columns:
                print(f"{col:<16} {'N/A':>8} {'N/A':>8}")
                continue
            vals = np.nan_to_num(df.iloc[te_idx][col].values.astype(float))
            auc = roc_auc_score(y_test, vals)
            if auc < 0.5:
                auc = roc_auc_score(y_test, -vals)
            f1 = best_f1(y_test, vals)
            agg[col]["auc"].append(auc)
            agg[col]["f1"].append(f1)
            print(f"{col:<16} {auc:>8.3f} {f1:>8.3f}")

        # CatBoost-5f
        X_test = df.iloc[te_idx][OPTIMAL_5F].astype(float)
        prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
        f1 = f1_score(y_test, (prob >= best_t).astype(int))
        agg["CatBoost-5f"]["auc"].append(auc)
        agg["CatBoost-5f"]["f1"].append(f1)
        print(f"{'CatBoost-5f':<16} {auc:>8.3f} {f1:>8.3f}")

    # 3. 平均表 (3 个测试集均值)
    print(f"\n{'='*60}")
    print("  平均 (3 个测试集)")
    print(f"{'='*60}")
    print(f"{'Method':<16} {'Avg_AUC':>10} {'Avg_F1':>10}")
    print("-"*38)
    for name in METHODS_ALL:
        aucs = agg[name]["auc"]
        f1s = agg[name]["f1"]
        if not aucs:
            print(f"{name:<16} {'N/A':>10} {'N/A':>10}")
            continue
        avg_auc = round(np.mean(aucs), 3)
        avg_f1 = round(np.mean(f1s), 3)
        print(f"{name:<16} {avg_auc:>10.3f} {avg_f1:>10.3f}")


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_features:
        print("="*60)
        print("Phase 1: 算 10 方法特征")
        print("="*60)
        compute_features(args)
    print("\n"+"="*60)
    print("Phase 2: 性能测试")
    print("="*60)
    evaluate()
