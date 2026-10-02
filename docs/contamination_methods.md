# Contamination Detection Methods: Principles and Settings

> 本文档描述实验中采用的 10 种数据污染检测方法的原理与具体设置。每个方法以一段学术性文字给出：核心直觉、计算方式、判定方向与默认阈值。

---

## 1. Zlib

Zlib 建立在"训练语料中的代码既具有低模型困惑度、又具有高压缩冗余"这一双重信号之上。对每条代码，先由目标模型计算其 token 序列的平均对数似然 `loss`（即平均 log-probability，通常为负值，越接近 0 表示模型越熟悉该代码），再对代码字节串进行 zlib 压缩得到 `compressed_len`，最终分数定义为 `zlib = loss / compressed_len`。被记忆的代码通常同时表现出较高的对数似然（loss 更接近 0）与较高的压缩率（重复模式多、`compressed_len` 更小），因此该比值更大（更接近 0）。判定方向为分数 **大于等于** 阈值 `-0.003` 时判为污染。

## 2. Min-K% (mink_0.2)

Min-K% Prob 基于"被训练代码中即使是最不可能出现的 token 也拥有异常高的概率"这一观察。方法先计算代码中每个 token 的对数概率，将其按升序排列，取概率最低的 20% 个 token 的对数概率均值作为分数。若模型在训练中见过该代码，则其尾部 token 的概率也会被抬高，导致该均值偏高。判定方向为分数 **大于等于** 阈值 `-3.0` 时判为污染。

## 3. Min-K%++ (mink++_0.2)

Min-K%++ 是 Min-K% 的改进版本，通过按序列自身的均值与方差对 token 对数概率做标准化，消除不同序列之间概率分布尺度差异带来的偏差，再取标准化后最低 20% 个 token 的均值。该归一化使分数在不同长度、不同风格的代码之间更具可比性。判定方向为分数 **大于等于** 阈值 `-3.0` 时判为污染。

## 4. DC-PDD

DC-PDD（Divergence Calibration with Probability Distribution Divergence）通过参考语料来校准目标模型对代码的似然，从而剥离"代码本身常见"这一混淆因素。方法首先在参考语料上构建平滑的 token 频率分布，然后对样本计算 token 级的对数概率，并在此基础上求取一种交叉熵型的偏差，取负均值作为分数。若目标模型对某代码的似然显著高于其"通用频率"所应给出的水平，则表明该代码被额外记忆。判定方向为分数 **小于等于** 阈值 `-3.0` 时判为污染。该方法除了目标模型外，还依赖一个参考模型（Qwen2.5-1.5B-Instruct）与一份独立的 Verilog 参考语料（`ref_verilog.jsonl`）：参考语料用于构建 token 频率分布，参考模型用于提供基准似然。

## 5. Perplexity (ppl)

PPL（Perplexity）采用目标模型对代码的似然作为污染信号。需要说明的是，尽管方法以"困惑度"命名，其分数字段实际上是代码 token 对数概率的均值（平均 log-probability，通常为负值），而非真正的困惑度 $\exp(-\bar\ell)$；二者单调负相关——对数概率越高（越接近 0），困惑度越低。被记忆的代码整体上具有更高的平均对数概率（等价于更低的困惑度），因此分数（平均 log-probability）更高。判定方向为分数 **大于等于** 阈值 `-2.2` 时判为污染。该分数与 Min-K% 共享同一组 token 对数概率，但取其整体均值而非尾部均值。

## 6. PETAL

PETAL（Probabilistic Estimation of Token likelihood via Alignment）是一种黑盒友好的检测方法，其核心是校准"语义相似度"对 logprob 的天然影响。方法先在代理模型上拟合一个"语义相似度 → token logprob"的线性映射，再在目标模型上计算代码内部 span 的语义相似度序列，并据此估计目标模型的 token logprob，最终分数为 `PETAL = -mean(estimated_log_probs)`。由于记忆带来的似然提升会超出"语义相似"所能解释的范围，因此该分数可识别记忆。判定方向为分数 **小于** 阈值 `0.8` 时判为污染。该方法依赖一个代理模型（Qwen2.5-1.5B-Instruct）与一个嵌入模型（all-MiniLM-L6-v2）。

## 7. Neighbor

Neighbor 通过比较目标代码与其"邻近变体"的似然来检测记忆：被记忆的代码应当比其近重复版本获得显著更高的似然。方法先用 CodeBERT 掩码语言模型与标识符一致性重命名生成若干邻居代码，再分别计算原代码与邻居代码的对数似然，取 `neighbor = ll_main - mean(ll_neighbors)` 作为分数。判定方向为分数 **大于等于** 阈值 `0.8` 时判为污染。

## 8. Reference

Reference 通过引入一个参考模型来对目标模型的似然做相对校准。其假设是：参考模型不太可能见过被检测的这些特定代码样本，因此其似然反映了一种"未记忆"的基线水平；若目标模型对某代码的似然显著高于参考模型，则表明目标模型对该代码存在额外的记忆。方法计算 `ref = ll_main - ll_ref_mean`，即目标模型对数似然减去参考模型对数似然的均值。判定方向为分数 **大于等于** 阈值 `0.13` 时判为污染。参考模型采用 Qwen2.5-1.5B-Instruct。

## 9. Token Overlap

Token Overlap 检测的是"模型能否根据前缀完整复现真实后缀"。方法首先将代码按比例切分为前缀（first piece）与真实后缀（second piece），然后让目标模型基于前缀生成补全：guided 提示在前缀之外附加该样本的 Instruction（模拟训练时的"指令 → 代码"任务），general 提示则只给前缀、作为"无 Instruction"的通用补全对照。用 ROUGE-L 分别度量两种补全与真实后缀的 token 重叠。切分比例取 `min_p = max_p = 20`（前缀占前 80%、真实后缀占后 20%），生成后缀的最大长度为 `max_new_tokens = 128`。其直觉是：若模型在训练中见过该代码（尤其是连同其 Instruction 一起见过），则给定前缀后它能准确地"回忆"出真实后缀，重叠显著偏高。最终取 guided 补全的相似度作为分数用于阈值判定；general 的相似度主要用于统计检验（检验 guided 是否显著高于 general，即 Instruction 是否确实有助于回忆），在本实验的下游判定中不直接参与。判定方向为分数 **大于等于** 阈值 `0.55` 时判为污染。

## 10. CDD

CDD（Contrastive Detection of Deviation）通过度量"模型生成的一致性/确定性"来检测记忆。其核心直觉是：被记忆的样本（模型在训练中见过"Instruction → 代码"的映射）在生成时高度确定，即便采用带随机性的采样解码，模型也会坚定地生成几乎相同的代码；而未被记忆的样本，模型不确定，采样会生成多种彼此不同的代码。

具体流程如下。给定样本的 Instruction，让目标模型生成代码，分别执行两类生成：其一是一次贪婪解码（greedy，每一步都取概率最高的 token，确定性最强），得到一个贪婪输出；其二是 `num_samples = 50` 次采样解码（按概率分布采样，`temperature = 0.8`、`top_p = 1.0`，每次结果不同），得到 50 个采样输出。随后对每个采样输出，计算它与贪婪输出之间的编辑距离；若编辑距离低于 `alpha × max_len`（`alpha = 0.05`，`max_len` 为被比较的 greedy 输出与 sampled 输出中较长者的 token 长度，即编辑距离低于代码长度的 5%），则视该采样输出与贪婪输出"一致"。最终 `contamination_score` 定义为"一致的采样输出占全部采样输出的比例"：若模型记住了该代码，采样输出几乎都与贪婪输出一致，比例接近 1；否则采样输出发散，比例接近 0。判定规则为 `contamination_score > xi`（`xi = 0.01`）时判为污染。该方法同时使用 Instruction 与 Response 作为输入，属于依赖指令的检测方法。

---

## 附：方法分类与输入依赖

| 方法 | 输入依赖 | 判定方向 | 阈值 |
|---|---|---|---|
| Zlib | Response（代码） | >= | -0.003 |
| Min-K% | Response | >= | -3.0 |
| Min-K%++ | Response | >= | -3.0 |
| DC-PDD | Response + 参考语料 | <= | -3.0 |
| Perplexity | Response | >= | -2.2 |
| PETAL | Response + 代理/嵌入模型 | < | 0.8 |
| Neighbor | Response + CodeBERT | >= | 0.8 |
| Reference | Response + 参考模型 | >= | 0.13 |
| Token Overlap | Response（guided 部分依赖 Instruction） | >= | 0.55 |
| CDD | Instruction + Response | > | 0.01 |

> 注：除 CDD 明确使用 Instruction 外，Token Overlap 的 guided 补全也依赖 Instruction；其余 8 种方法仅使用 Response（代码）字段。这一输入依赖差异决定了当负样本 Instruction 重新生成时，只有 CDD 与 Token Overlap 的特征需要重算。
