# 10 种污染检测方法的第三方脚本

本目录存放 10 种污染检测方法的**第三方实现脚本**，不随仓库提供，请按 README 第 1 节从对应仓库获取后放置。

`run_3new_10probes.py` 按以下相对路径引用这些脚本：

```
scripts/10_methods/
├── Mink++/
│   ├── run_zlib_mink_mink++.py
│   ├── run_neighbor.py
│   └── run_ref.py
├── DCPDD/
│   ├── run_local_dcpdd_ppl_eval.py
│   └── ref_code/ref_verilog.jsonl        # DCPDD 参考语料
├── PETAL/
│   └── run.py
├── data-contamination-main/
│   └── code/Token_Overlap/run_local_code_overlap_eval.py
└── CDD/
    └── run_local_cdd_eval.py
```

| 方法 | 脚本 | 来源仓库 |
|---|---|---|
| zlib / Min-K% / Min-K%++ | `Mink++/run_zlib_mink_mink++.py` | Mink++ |
| Neighbor | `Mink++/run_neighbor.py` | Mink++ |
| Reference | `Mink++/run_ref.py` | Mink++ |
| DC-PDD / PPL | `DCPDD/run_local_dcpdd_ppl_eval.py` | DCPDD |
| PETAL | `PETAL/run.py` | PETAL |
| Token Overlap | `data-contamination-main/code/Token_Overlap/run_local_code_overlap_eval.py` | data-contamination |
| CDD | `CDD/run_local_cdd_eval.py` | CDD-TED4LLMs |

每个方法的原理、阈值、输入依赖见 `docs/contamination_methods.md`。
