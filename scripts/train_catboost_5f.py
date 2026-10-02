#!/usr/bin/env python3
"""
重训 CatBoost-5f (C+O, 最优 5f, none 归一化), 输出新 τ* + OOF F1, 保存模型与阈值。

配置与之前完全一致:
  - 训练源: CodeV + OriGen (C+O)
  - 特征: 最优 5f = DCPDD, Reference, neighbor, mink_0.2, zlib
  - CatBoost: iterations=900, depth=6, lr=0.03, Logloss, AUC, seed=42
  - OOF 阈值: 5 折分层 CV, grid 0.05~0.95 step 0.005, 最大化 F1
  - 训练集 = all_1000 去掉 test_200 (即 train_800)

用法(服务器):
  python train_catboost_5f.py
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

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FEAT_DIR = DATA_DIR / "features" / "golden"     # CodeV/OriGen 的 10 方法特征
TRAIN_GOLDEN_DIR = DATA_DIR / "golden"                 # golden 数据集

OUT_MODEL = DATA_DIR / "features" / "golden" / "catboost_5f.cbm"
OUT_THRESH = DATA_DIR / "features" / "golden" / "catboost_5f_threshold.json"


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


def main():
    print("加载 C+O 特征...")
    train_feats = {}; train_labels = {}
    for m in ["CodeV", "OriGen"]:
        df, lb = load_features(TRAIN_FEAT_DIR / m)
        if df is None:
            print(f"  [ERROR] {m} 特征加载失败")
            return
        train_feats[m] = df; train_labels[m] = lb
        print(f"  {m}: {len(df)} 样本, 特征列 {list(df.columns)}")

    # 分层分割: all 去掉 test_200
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
        print(f"  {m}: train={len(tr_idx)} test={len(te_idx)}")

    X_train = pd.concat([train_feats["CodeV"].iloc[splits["CodeV"]["train"]],
                         train_feats["OriGen"].iloc[splits["OriGen"]["train"]]],
                        ignore_index=True)[OPTIMAL_5F].astype(float)
    y_train = np.concatenate([train_labels["CodeV"][splits["CodeV"]["train"]],
                              train_labels["OriGen"][splits["OriGen"]["train"]]])
    print(f"\nX_train={X_train.shape}, 正样本={int(y_train.sum())}, 负样本={int((1-y_train).sum())}")

    # 训练完整模型
    print("\n训练 CatBoost-5f ...")
    model = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.03,
                               loss_function="Logloss", eval_metric="AUC",
                               random_seed=42, verbose=False, allow_writing_files=False)
    model.fit(X_train, y_train)

    # 5 折 OOF
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

    print("\n" + "="*50)
    print(f"  OOF 阈值  τ* = {best_t:.4f}")
    print(f"  OOF F1       = {best_f:.4f}")
    print("="*50)

    # 保存模型 + 阈值
    model.save_model(str(OUT_MODEL))
    with open(OUT_THRESH, "w", encoding="utf-8") as f:
        json.dump({"threshold": best_t, "oof_f1": best_f,
                   "optimal_5f": OPTIMAL_5F, "n_train": int(len(X_train))}, f, indent=2)
    print(f"\nSaved model -> {OUT_MODEL}")
    print(f"Saved threshold -> {OUT_THRESH}")


if __name__ == "__main__":
    main()
