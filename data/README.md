# data 目录说明

本目录存放数据，请按以下结构放置：

```
data/
├── benchmarks/          # 6 个 benchmark 数据 (VerilogEval_v1/v2, RTLLM_v1/v2, RealBench_module/system)
├── golden/              # CodeV/RTLCoder/OriGen 的 golden 数据 (train_800 / test_200)
├── golden_haven/        # HAVEN/HAVEN_CodeQwen/vericoder 的 golden 数据 (test_200)
├── features/            # 10 方法特征分数 (需自行用第三方方法脚本计算)
│   ├── golden/          #   CodeV/OriGen 特征
│   ├── haven/           #   HAVEN 批特征
│   └── benchmark/       #   benchmark 特征
└── chapter3_probes_v2/  # 5 模型 × 6 benchmark 的 10 方法特征 (含 overlap_matrix.csv)
```

> 数据文件与特征分数不随仓库提供，请按 README 的"复现步骤"准备。
