# Verilog 代码生成模型的数据污染检测

本项目实现了对 Verilog 代码生成模型（CodeGen、RTLCoder、OriGen、ChipGPT 等）的**训练数据污染检测**：采用 10 种污染检测方法 + 一个 CatBoost 集成探测器，检测多个 benchmark 的参考代码是否出现在这些模型的训练数据中。

---

## 目录结构

```
repo/
├── scripts/                    # 核心脚本（直接放本目录，见下方清单）
│   └── 10_methods/             # 10 种污染检测方法（第三方，需另装）
├── data/                       # 数据（见下方清单）
│   ├── benchmarks/             # 6 个 benchmark 数据
│   ├── golden1/                # golden 数据集（CodeV/RTLCoder/OriGen）
│   ├── golden2/                # golden 数据集（HAVEN/HAVEN_CodeQwen/vericoder）
│   └── features/               # 10 方法特征分数
├── docs/                       # 方法与原理说明（md/tex）
├── models/                     
└── README.md
```


---

### 1. 10 种污染检测方法

检测方法来自以下第三方实现，需要单独安装/放置：

| 方法 | 脚本 | 来源 |
|---|---|---|
| zlib / Min-K% / Min-K%++ | `run_zlib_mink_mink++.py` | Mink++ 仓库 |
| Neighbor | `run_neighbor.py` | Mink++ 仓库 |
| Reference | `run_ref.py` | Mink++ 仓库 |
| DC-PDD / PPL | `run_local_dcpdd_ppl_eval.py` | DCPDD 仓库 |
| PETAL | `run.py` | PETAL 仓库 |
| Token Overlap | `run_local_code_overlap_eval.py` | data-contamination 仓库 |
| CDD | `run_local_cdd_eval.py` | CDD-TED4LLMs 仓库 |

每个方法的原理、阈值、输入依赖见 `docs/contamination_methods.md`。

### 2. 6 个 benchmark 数据

| benchmark | 题数 | 说明 |
|---|---|---|
| VerilogEval_v1 | 156 | Human 子集，Instruction = 文字描述 + 模块接口 |
| VerilogEval_v2 | 156 | spec-to-rtl 子集，Instruction = 纯文字描述 |
| RTLLM_v1 | 29 | v1.0 |
| RTLLM_v2 | 50 | v2.0 |
| RealBench_module | 60 | 模块级 |
| RealBench_system | 4 | 系统级 |


### 3. 5 模型 × 6 benchmark 的 10 方法不一致实验

- 脚本：`analyze_overlap_heatmap.py`
- 输出：各方法的污染比例表、方法间 `overlap_matrix.csv`

### 4. 重叠热力图

- 脚本：`plot_overlap_heatmap_server.py`
- 输入：`overlap_matrix.csv`
- 输出：10 方法共同污染样本数热力图

### 5. golden 数据集（数据，非脚本）

- **CodeV / RTLCoder / OriGen** 的训练集 `train_800` 与测试集 `test_200`（各 400+400 / 100+100）
- **HAVEN / HAVEN_CodeQwen / vericoder** 的测试集 `test_200`（各 100+100）


### 6. 训练探测器

- 脚本：`train_catboost_5f.py`
- 训练：CatBoost（最优 5 特征：DCPDD / Reference / neighbor / mink_0.2 / zlib），训练源 CodeV + OriGen
- 输出：模型 `catboost_5f.cbm` + 阈值 `τ*`

### 7. CodeV / RTLCoder / OriGen 测试集上的性能

- 脚本：`final_vs_baselines_5f.py`
- 对比：10 单方法 vs CatBoost-5f 的 AUC / F1

### 8. HAVEN / HAVEN_CodeQwen / vericoder 测试集上的性能

- 脚本：`run_3new_10probes.py`
- 对比：10 单方法 vs CatBoost-5f 的 AUC / F1

### 9. 9 模型 × 6 benchmark 污染率探测

- 脚本：`run_optimal_detect_9models.py`
- 用训练好的探测器，检测 9 个模型在 6 个 benchmark 上的污染率

### 10. 山脊图实验

- 脚本：`plot_violin_contamination.py`
- 输出：9 个模型的污染概率分布山脊图（ridgeline）

### 11. Lorenz 图实验

- 脚本：`plot_lorenz_contamination.py`
- 输出：污染集中度 Lorenz 偏差曲线 + Gini 系数

### 12. Boosting 算法选择消融

- 脚本：`ablation_boost_methods_v2.py`
- 对比：CatBoost / XGBoost / LightGBM

### 13. 探测器特征分数贪心选择消融

- 脚本：`ablation_greedy_dcpdd.py`
- 从 DCPDD 出发，逐步加入使 Avg AUC 提升最大的特征

### 14. 训练源和训练数据数量消融

- 脚本：`ablation_train_size_v2.py`
- 分析：单源/双源/三源 × 训练量（tc=50→400）

---

## 路径配置说明

所有脚本**不再硬编码绝对路径**，而是在脚本开头通过 `BASE_DIR` 定位项目根目录，再用相对路径引用数据：

```python
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent   # 项目根目录

DATA_DIR     = BASE_DIR / "data"
GOLDEN_DIR   = DATA_DIR / "golden"                   # CodeV/RTLCoder/OriGen 的 golden 数据
GOLDEN_HAVEN = DATA_DIR / "golden_haven"             # HAVEN 批的 golden 数据
BENCH_DIR    = DATA_DIR / "benchmarks"               # 6 个 benchmark 数据
FEAT_DIR     = DATA_DIR / "features"                 # 特征分数
MODEL_DIR    = BASE_DIR / "models"                   # 模型权重
```

运行时把对应数据放到上述目录即可，无需修改脚本。

---

- ## 模型
- 
- 需要自行下载的模型（放入 `models/`）：
- 
- | 模型 | 来源 |
- |---|---|
- | CodeGen-6B-Multi | https://huggingface.co/Salesforce/codegen-6B-multi |
- | Phi-2 | https://huggingface.co/microsoft/phi-2 |
- | DeepSeek-Coder-6.7B | https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct |
- | Qwen2.5-Coder-7B | https://huggingface.co/Qwen/Qwen2.5-Coder-7B |
- | ChipGPT | https://www.modelscope.cn/models/changkaiyan/ChipGPT-Llama2-SFT-7B |
- | RTLCoder | https://huggingface.co/ishorn5/RTLCoder-Deepseek-v1.1 |
- | VeriGen | https://huggingface.co/shailja/fine-tuned-codegen-6B-Verilog |
- | Qwen2.5-1.5B-Instruct（参考模型） | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
- | CodeBERT | https://huggingface.co/microsoft/codebert-base |
- | all-MiniLM-L6-v2 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
- | OriGen | https://huggingface.co/henryen/OriGen |
- | VeriCoder | https://huggingface.co/LLM4Code/VeriCoder_Qwen14B |


## 复现步骤

1. 下载/放置 10 种检测方法的第三方脚本（见第 1 节）；
2. 下载模型权重到 `models/`（见"模型"）；
3. 放置 benchmark 数据到 `data/benchmarks/`、golden 数据到 `data/golden/` 与 `data/golden_haven/`；
4. 使用第三方方法脚本对 golden 与 benchmark 样本计算 10 方法特征，放入 `data/features/`（特征计算 pipeline 脚本不公开）；
5. 运行 `train_catboost_5f.py` 训练探测器；
6. 运行性能对比、污染率探测、消融、画图脚本。

---


