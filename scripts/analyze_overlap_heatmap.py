#!/usr/bin/env python3
"""第3章 v2 分析：污染比例表 + 全局编号 + 方法间重叠热力图"""
import json, argparse, numpy as np, pandas as pd
from pathlib import Path
from typing import List, Tuple

# 10 个方法: (显示名, 子目录, score字段, 阈值, 方向)
METHOD_SPECS: List[Tuple[str, str, str, float, str]] = [
    ("Zlib",          "mink_family",  "zlib",               -0.003, ">="),
    ("Min-K%",        "mink_family",  "mink_0.2",           -3.0,   ">="),
    ("Min-K%++",      "mink_family",  "mink++_0.2",         -3.0,   ">="),
    ("DC-PDD",        "dcpdd_ppl",    "DCPDD",              -3.0,   "<="),
    ("PPL",           "dcpdd_ppl",    "PPL",                -2.2,   ">="),
    ("PETAL",         "petal",        "PETAL",              0.8,    "<"),
    ("Neighbor",      "neighbor",     "neighbor",           0.8,    ">="),
    ("Reference",     "reference",    "ref",                0.13,   ">="),
    ("Overlap",       "token_overlap","contamination_score", 0.55,  ">="),
    ("CDD",           "cdd",          "contamination_score", 0.01,   ">"),
]

MODELS = ["ChipGPT", "RTLCoder", "OriGen", "VeriCoder", "VeriGen"]
BENCHMARKS = ["VerilogEval_v1", "VerilogEval_v2", "RTLLM_v1", "RTLLM_v2",
              "RealBench_module", "RealBench_system"]


def apply_threshold(score, th, direction):
    if direction == ">=":
        return 1 if score >= th else 0
    elif direction == ">":
        return 1 if score > th else 0
    elif direction == "<=":
        return 1 if score <= th else 0
    elif direction == "<":
        return 1 if score < th else 0
    return 0


def load_scores(results_dir, score_field):
    path = Path(results_dir) / "results.jsonl"
    if not path.exists():
        return []
    scores = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            try:
                scores.append(float(r.get(score_field, np.nan)))
            except (ValueError, TypeError):
                scores.append(np.nan)
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default=str(Path(__file__).resolve().parent.parent / "data" / "chapter3_probes_v2"))
    args = p.parse_args()
    root = Path(args.output_root)

    method_names = [m[0] for m in METHOD_SPECS]
    n_methods = len(method_names)

    # 每个方法收集"检测为污染"的全局样本编号集合
    contaminated_sets = {m: set() for m in method_names}
    table_rows = []

    for model in MODELS:
        for bench in BENCHMARKS:
            pair_dir = root / model / bench
            if not pair_dir.exists():
                continue
            for mname, subdir, field, th, direction in METHOD_SPECS:
                d = pair_dir / subdir
                scores = load_scores(d, field)
                if not scores:
                    continue
                preds = [apply_threshold(s, th, direction) for s in scores]
                n_total = len(scores)
                n_contam = sum(preds)
                ratio = round(n_contam / n_total, 3) if n_total else np.nan

                table_rows.append({
                    "model": model, "benchmark": bench, "method": mname,
                    "n_total": n_total, "n_contaminated": n_contam,
                    "contamination_ratio": ratio,
                })

                # 全局编号 + 收集污染样本
                for local_idx, pr in enumerate(preds):
                    if pr == 1:
                        global_id = f"{model}|{bench}|{local_idx}"
                        contaminated_sets[mname].add(global_id)

    # ---- 1. 污染比例表 ----
    df = pd.DataFrame(table_rows)
    csv_path = root / "contamination_ratio_table.csv"
    df.to_csv(csv_path, index=False)
    pivot = df.pivot_table(index=["model", "benchmark"], columns="method", values="contamination_ratio")
    pivot_path = root / "contamination_ratio_pivot.csv"
    pivot.to_csv(pivot_path)
    print(f"[OK] 比例表 -> {csv_path}")
    print(f"[OK] 透视表 -> {pivot_path}")

    # ---- 2. 打印各方法检测为污染的样本总数 ----
    print("\n各方法检测为污染的样本总数:")
    for m in method_names:
        print(f"  {m:<16}: {len(contaminated_sets[m])}")

    # ---- 3. 方法间重叠矩阵 (10x10) ----
    overlap = np.zeros((n_methods, n_methods), dtype=int)
    for i, mi in enumerate(method_names):
        for j, mj in enumerate(method_names):
            overlap[i, j] = len(contaminated_sets[mi] & contaminated_sets[mj])

    overlap_df = pd.DataFrame(overlap, index=method_names, columns=method_names)
    overlap_df.to_csv(root / "overlap_matrix.csv")
    print(f"\n[OK] 重叠矩阵 -> {root / 'overlap_matrix.csv'}")
    print(overlap_df)

    # ---- 4. 热力图 (学术风格) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'stix',
        'font.size': 10,
        'axes.labelsize': 11,
    })

    fig, ax = plt.subplots(figsize=(10.5, 9))
    mask = np.triu(np.ones_like(overlap, dtype=bool), k=1)  # 只显示下三角 + 对角线

    sns.heatmap(overlap, annot=False, cmap="Blues",
                xticklabels=method_names, yticklabels=method_names,
                mask=mask, square=True, linewidths=0,
                cbar_kws={"label": "Co-contaminated samples", "shrink": 0.8},
                vmin=0, ax=ax)

    # 手动标注下三角(含对角线): 对角线白字粗体, 非对角线黑字
    for i in range(n_methods):
        for j in range(i + 1):
            val = overlap[i, j]
            if i == j:
                ax.text(j + 0.5, i + 0.5, f"{val:d}",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
            else:
                ax.text(j + 0.5, i + 0.5, f"{val:d}",
                        ha="center", va="center", fontsize=8.5,
                        color="black")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # 去掉顶部和右侧边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    plt.tight_layout()
    fig.savefig(root / "overlap_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / "overlap_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 热力图 -> {root / 'overlap_heatmap.png'} / .pdf")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
