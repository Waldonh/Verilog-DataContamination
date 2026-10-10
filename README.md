# Data Contamination Detection for Verilog Code Generation Models

This project implements **training data contamination detection** for Verilog code generation models (CodeGen, RTLCoder, OriGen, ChipGPT, etc.): it uses 10 contamination detection methods + a CatBoost ensemble detector to detect whether the reference code of multiple benchmarks appears in the training data of these models.

---

## Directory Structure

```
repo/
├── scripts/                    # Core scripts (placed directly in this directory; see the list below)
│   └── 10_methods/             # 10 contamination detection methods (third-party, need separate installation)
├── data/                       # Data (see the list below)
│   ├── benchmarks/             # Data for 6 benchmarks
│   ├── golden1/                # Golden datasets (CodeV/RTLCoder/OriGen)
│   ├── golden2/                # Golden datasets (HAVEN/HAVEN_CodeQwen/vericoder)
│   └── features/               # Feature scores of the 10 methods
├── docs/                       # Method and principle descriptions (md/tex)
├── models/                     
└── README.md
```

---

### 1. 10 Contamination Detection Methods

The detection methods come from the following third-party implementations and need to be installed/placed separately:

| Method | Script | Source |
|---|---|---|
| zlib / Min-K% / Min-K%++ | `run_zlib_mink_mink++.py` | Mink++ repository |
| Neighbor | `run_neighbor.py` | Mink++ repository |
| Reference | `run_ref.py` | Mink++ repository |
| DC-PDD / PPL | `run_local_dcpdd_ppl_eval.py` | DCPDD repository |
| PETAL | `run.py` | PETAL repository |
| Token Overlap | `run_local_code_overlap_eval.py` | data-contamination repository |
| CDD | `run_local_cdd_eval.py` | CDD-TED4LLMs repository |

For the principle, threshold, and input dependencies of each method, see `docs/contamination_methods.md`.

### 2. Data for 6 Benchmarks

| Benchmark | No. of Problems | Description |
|---|---|---|
| VerilogEval_v1 | 156 | Human subset, Instruction = textual description + module interface |
| VerilogEval_v2 | 156 | spec-to-rtl subset, Instruction = plain textual description |
| RTLLM_v1 | 29 | v1.0 |
| RTLLM_v2 | 50 | v2.0 |
| RealBench_module | 60 | module-level |
| RealBench_system | 4 | system-level |

### 3. 10-Method Disagreement Experiment on 5 Models × 6 Benchmarks

- Script: `analyze_overlap_heatmap.py`
- Output: contamination ratio table for each method, inter-method `overlap_matrix.csv`

### 4. Overlap Heatmap

- Script: `plot_overlap_heatmap_server.py`
- Input: `overlap_matrix.csv`
- Output: heatmap of the number of samples jointly contaminated by the 10 methods

### 5. Golden Datasets (Data, Not Scripts)

- **CodeV / RTLCoder / OriGen** training set `train_800` and test set `test_200` (400+400 / 100+100 each)
- **HAVEN / HAVEN_CodeQwen / vericoder** test set `test_200` (100+100 each)

### 6. Training the Detector

- Script: `train_catboost_5f.py`
- Training: CatBoost (optimal 5 features: DCPDD / Reference / neighbor / mink_0.2 / zlib), training sources: CodeV + OriGen
- Output: model `catboost_5f.cbm` + threshold `τ*`

### 7. Performance on CodeV / RTLCoder / OriGen Test Sets

- Script: `final_vs_baselines_5f.py`
- Comparison: 10 individual methods vs CatBoost-5f in terms of AUC / F1

### 8. Performance on HAVEN / HAVEN_CodeQwen / vericoder Test Sets

- Script: `run_3new_10probes.py`
- Comparison: 10 individual methods vs CatBoost-5f in terms of AUC / F1

### 9. Contamination Rate Detection on 9 Models × 6 Benchmarks

- Script: `run_optimal_detect_9models.py`
- Use the trained detector to detect the contamination rate of 9 models on 6 benchmarks

### 10. Ridgeline Plot Experiment

- Script: `plot_violin_contamination.py`
- Output: ridgeline plot of contamination probability distributions for the 9 models

### 11. Lorenz Plot Experiment

- Script: `plot_lorenz_contamination.py`
- Output: Lorenz deviation curve of contamination concentration + Gini coefficient

### 12. Boosting Algorithm Selection Ablation

- Script: `ablation_boost_methods_v2.py`
- Comparison: CatBoost / XGBoost / LightGBM

### 13. Greedy Feature-Score Selection Ablation for the Detector

- Script: `ablation_greedy_dcpdd.py`
- Starting from DCPDD, greedily add the feature that gives the largest improvement in Avg AUC at each step

### 14. Ablation on Training Sources and Training Data Size

- Script: `ablation_train_size_v2.py`
- Analysis: single-source / dual-source / triple-source × training size (tc=50→400)

---

## Path Configuration Instructions

All scripts **no longer hard-code absolute paths**. Instead, at the beginning of each script, they locate the project root directory via `BASE_DIR`, and then reference data using relative paths:

```python
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent   # Project root directory

DATA_DIR     = BASE_DIR / "data"
GOLDEN_DIR   = DATA_DIR / "golden"                   # Golden data for CodeV/RTLCoder/OriGen
GOLDEN_HAVEN = DATA_DIR / "golden_haven"             # Golden data for the HAVEN batch
BENCH_DIR    = DATA_DIR / "benchmarks"               # Data for the 6 benchmarks
FEAT_DIR     = DATA_DIR / "features"                 # Feature scores
MODEL_DIR    = BASE_DIR / "models"                   # Model weights
```

At runtime, simply place the corresponding data into the above directories; no script modification is needed.

---

## Models

Models that need to be downloaded separately (place into `models/`):

| Model | Source |
|---|---|
| CodeGen-6B-Multi | https://huggingface.co/Salesforce/codegen-6B-multi |
| Phi-2 | https://huggingface.co/microsoft/phi-2 |
| DeepSeek-Coder-6.7B | https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct |
| Qwen2.5-Coder-7B | https://huggingface.co/Qwen/Qwen2.5-Coder-7B |
| ChipGPT | https://www.modelscope.cn/models/changkaiyan/ChipGPT-Llama2-SFT-7B |
| RTLCoder | https://huggingface.co/ishorn5/RTLCoder-Deepseek-v1.1 |
| VeriGen | https://huggingface.co/shailja/fine-tuned-codegen-6B-Verilog |
| Qwen2.5-1.5B-Instruct (reference model) | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
| CodeBERT | https://huggingface.co/microsoft/codebert-base |
| all-MiniLM-L6-v2 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| OriGen | https://huggingface.co/henryen/OriGen |
| VeriCoder | https://huggingface.co/LLM4Code/VeriCoder_Qwen14B |

## Reproduction Steps

1. Download/place the third-party scripts for the 10 detection methods (see Section 1);
2. Download model weights into `models/` (see "Models");
3. Place benchmark data into `data/benchmarks/`, and golden data into `data/golden/` and `data/golden_haven/`;
4. Use the third-party method scripts to compute the 10-method features for the golden and benchmark samples, and place them into `data/features/` (the feature computation pipeline scripts are not public);
5. Run `train_catboost_5f.py` to train the detector;
6. Run the performance comparison, contamination rate detection, ablation, and plotting scripts.

---
