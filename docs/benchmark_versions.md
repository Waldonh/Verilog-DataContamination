# Benchmark Selection and Versions

> 本文档说明实验中采用的 6 个 benchmark 的**来源、版本、题数、任务形式**，以及检测时使用的 Instruction（prompt）与 Response（参考代码）的具体形式。

---

## 概览

| Benchmark | 来源/版本 | 题数 | 任务形式 | Instruction 形式 |
|---|---|---|---|---|
| VerilogEval_v1 | VerilogEval release 1.0.0（Human 子集） | 156 | spec-to-rtl | 文字描述 + 模块接口代码 |
| VerilogEval_v2 | VerilogEval release 2.0.0（spec-to-rtl 子集） | 156 | spec-to-rtl | 纯文字描述 |
| RTLLM_v1 | RTLLM v1.0 | 29 | spec-to-rtl | 文字描述 + 端口列表（Verilog 语法风格） |
| RTLLM_v2 | RTLLM v2.0 | 50 | spec-to-rtl | 文字描述 + 端口列表（文字描述风格） |
| RealBench_module | RealBench | — | — | — |
| RealBench_system | RealBench | — | — | — |

---

## 1. VerilogEval v1

**来源与版本**：VerilogEval release 1.0.0，取 **Human 子集**（156 题，源自 HDLBits 的人工描述题目）。

**任务形式**：spec-to-rtl（给定规格描述，生成完整 RTL）。Human 子集原始数据中，`prompt` 仅含模块接口代码（`module top_module(...)`），文字描述存放在独立的 `descriptions/VerilogDescription_Human.jsonl` 的 `detail_description` 字段中。本实验将二者拼接，得到完整 Instruction：

> `Instruction` = `detail_description`（文字描述）+ 空行 + 模块接口代码

**Response（参考代码）**：`canonical_solution`（HDLBits 的标准答案，模块名为 `top_module`）。

---

## 2. VerilogEval v2

**来源与版本**：VerilogEval release 2.0.0（ICCAD 2023），取 **spec-to-rtl 子集**（156 题）。

**任务形式**：spec-to-rtl。该子集是本实验最终采用的版本，其 Instruction 为**纯文字描述**（包含模块名、端口文字说明与功能描述，不含任何模块接口代码）：

> `Instruction` = "I would like you to implement a module named TopModule ..."

**Response（参考代码）**：`_ref.sv`（参考代码，模块名为 `RefModule`）。

> 注：VerilogEval v2 另有一个 code-complete 子集（156 题），其 Instruction 含模块接口代码、任务为代码补全。本实验统一采用 spec-to-rtl，与 v1 的 Human 子集保持任务类型一致（均为"规格 → RTL"）。v1 Human 与 v2 的两批 156 题源自同一批 HDLBits 题目，但 v1 的 canonical_solution 与 v2 的 ref.sv 写法不同（模块名分别为 `top_module` 与 `RefModule`）。

---

## 3. RTLLM v1

**来源与版本**：RTLLM v1.0（ASP-DAC 2024 论文对应的 release），共 **29 个设计**。

**任务形式**：spec-to-rtl。Instruction 为文字描述 + 端口列表，端口以 **Verilog 语法风格**给出（带 `input`/`output` 关键词与位宽）：

> `Instruction` = "Please act as a professional verilog designer. ..." + 模块名 + `input [31:0] a,` 形式的端口列表

**Response（参考代码）**：`verified_*.v`（人工验证过的正确代码，模块名为 `verified_<design>`）。

---

## 4. RTLLM v2

**来源与版本**：RTLLM v2.0，共 **50 个设计**（Arithmetic 19 + Control 6 + Memory 5 + Miscellaneous 20）。

**任务形式**：spec-to-rtl。Instruction 为文字描述 + 端口列表，端口以 **文字描述风格**给出（端口名 + 位宽 + 一句说明，不含 `input`/`output` 关键词）：

> `Instruction` = "Please act as a professional verilog designer. ..." + 模块名 + `a[15:0]: 16-bit input operand A.` 形式的端口列表

**Response（参考代码）**：`verified_*.v`（人工验证过的正确代码）。

> 注：RTLLM v2 复用 v1 的设计时采用了 v1.1 修复版的 description（文字描述风格端口、修复了 v1.0 的笔误）；RTLLM v1 则保持 v1.0 原版（Verilog 语法风格端口）。因此二者端口列表风格不完全一致，均沿用各自官方 release 的原始格式。

---

## 5. RealBench

**来源**：RealBench，取 `RealBench_module` 与 `RealBench_system` 两个子集，分别对应模块级与系统级的 RTL 设计任务。

---

## 关键约定

- 所有 benchmark 的 **Response 字段均为参考代码（黄金答案）**（`canonical_solution` / `_ref.sv` / `verified_*.v`），而非 LLM 生成代码；
- 污染检测即判断这些参考代码是否出现在各模型的训练数据中；
- 除 CDD 明确使用 Instruction、Token Overlap 的 guided 部分依赖 Instruction 外，其余 8 种方法仅使用 Response（代码）。
