# Contamination across RTL Code-Generation Models and Benchmarks

> **中译**：RTL 代码生成模型与 benchmark 之间的污染

> 本文件详细说明两个分析图（山脊图、Lorenz 图）以及前置污染率表的**数据如何计算**、**图的目的是什么**、**能得出哪些结论**。每节为学术化英文 + 逐段中文翻译，具体数值用 `[ ]` 占位。

---

# 0. 前置：污染概率 p 是如何算出来的（两个图的共同数据基础）

两个图的原始输入，都是"每条 benchmark 样本的污染概率 $p$"。要理解图，必须先理解 $p$ 从何而来。整条链路分四步。

**中译**：两个图的原始输入都是"每条 benchmark 样本的污染概率 $p$"。理解图之前须先理解 $p$ 的来历。整条链路分四步。

## 0.1 五个探针特征的计算

**EN:** For each code sample, five probe features are computed, each capturing a distinct statistical signature of training-data exposure:

- **zlib**: the model's mean token log-loss divided by the zlib-compressed length of the code, $z = \text{loss}/\text{compressed\_len}$. Memorized code tends to be both low-loss and highly compressible (repetitive), so the ratio separates seen from unseen code.
- **mink_0.2 (Min-K% Prob)**: token log-probabilities are sorted in ascending order and the mean of the lowest 20% is taken. Memorized sequences have unusually high probabilities even on their *least* likely tokens.
- **DCPDD**: builds a smoothed token-frequency distribution from a reference corpus, then computes a cross-entropy-style divergence between the sample's token probabilities and this reference, taken with a negative sign.
- **Reference**: the difference between the target model's log-likelihood of the code and a reference model's log-likelihood, $\text{ref} = \text{ll}_\text{main} - \text{ll}_\text{ref}$. A model that has seen the code assigns it disproportionately higher likelihood than a model that has not.
- **neighbor**: the target model's log-likelihood of the code minus the mean log-likelihood of its *neighbors* (perturbed variants produced by CodeBERT masked-language modeling and identifier renaming), $\text{nb} = \text{ll}_\text{main} - \text{mean}(\text{ll}_\text{neighbors})$. Memorized code stands out sharply against its near-duplicates.

**中译**：对每条代码样本，计算五个探针特征，各自刻画训练数据暴露的一种统计特征：

- **zlib**：模型对代码的平均 token log-loss 除以代码的 zlib 压缩长度，$z=\text{loss}/\text{compressed\_len}$。被记忆的代码往往既低 loss 又高度可压缩（重复模式多），因此该比值能区分见过与未见过。
- **mink_0.2（Min-K% Prob）**：将 token log-probability 升序排列，取最低 20% 的均值。被记忆的序列即使在其"最不可能"的 token 上也拥有异常高的概率。
- **DCPDD**：由参考语料构建平滑的 token 频率分布，再计算样本 token 概率与该参考之间的交叉熵型偏差，并取负号。
- **Reference**：目标模型对代码的 log-likelihood 减去参考模型的 log-likelihood，$\text{ref}=\text{ll}_\text{main}-\text{ll}_\text{ref}$。见过该代码的模型会赋予它不成比例更高的似然。
- **neighbor**：目标模型对代码的 log-likelihood 减去其"邻居"（由 CodeBERT 掩码语言模型与标识符重命名产生的扰动变体）log-likelihood 的均值，$\text{nb}=\text{ll}_\text{main}-\text{mean}(\text{ll}_\text{neighbors})$。被记忆的代码会显著高于其近重复变体。

## 0.2 CatBoost 融合与污染概率 p 的输出

**EN:** These five features are fed into a CatBoost classifier trained on labeled CodeV and OriGen data (positive = contaminated, negative = clean), with no normalization. The classifier outputs a scalar contamination probability $p \in [0,1]$ via `predict_proba`, where $p$ is the probability of the positive (contaminated) class. Thus each sample is reduced from five heterogeneous probe scores to a single, calibrated probability.

**中译**：这五个特征输入一个 CatBoost 分类器，该分类器用带标签的 CodeV 与 OriGen 数据（正 = 污染、负 = 干净）训练，且不做归一化。分类器通过 `predict_proba` 输出标量污染概率 $p\in[0,1]$，其中 $p$ 为正类（污染）的概率。于是每条样本从五个异质探针分数被压缩为一个校准后的概率。

## 0.3 决策阈值 τ* 的确定

**EN:** The decision threshold $\tau^*$ is chosen by 5-fold stratified cross-validation: out-of-fold predictions are collected on the training set, and the threshold maximizing F1 over a grid in $[0.05, 0.95]$ is selected ($\tau^* \approx 0.405$). This threshold is used only for binary decisions (e.g., computing contamination *rates*), not for the two distribution figures, which use the raw $p$ directly.

**中译**：决策阈值 $\tau^*$ 由 5 折分层交叉验证确定：在训练集上收集 out-of-fold 预测，在 $[0.05,0.95]$ 的网格上选取使 F1 最大的阈值（$\tau^*\approx0.405$）。该阈值仅用于二值决策（如计算污染率），两个分布图不依赖它，而是直接使用原始 $p$。

## 0.4 对 benchmark 打分与按模型汇总

**EN:** For each of the nine models and each of the six benchmarks, the five probe features are computed for every benchmark sample and passed through the trained classifier, yielding a per-sample $p$. The samples carry no ground-truth labels (benchmarks are real test data), so no correctness judgment is made—we only observe the *distribution* of $p$. Finally, for each model, the $p$ values of all its samples across all six benchmarks are pooled into a single collection. Each model therefore contributes one probability distribution, which is the raw material of both figures.

**中译**：对 9 个模型 × 6 个 benchmark 的每一对，计算每条 benchmark 样本的五个探针特征并送入训练好的分类器，得到逐样本的 $p$。这些样本没有真实标签（benchmark 是真实测试数据），因此不做正确性判断，只观察 $p$ 的分布。最后，对每个模型，把它在全部 6 个 benchmark 上所有样本的 $p$ 汇总为一个集合。每个模型由此贡献一条概率分布，这正是两个图的原始素材。

---

# 1. 山脊图（Ridgeline Plot）

**中译**：山脊图

## 1.1 数据是怎么算出来的

**EN:** For each model's pooled collection of $p$ values, the following steps are applied:

1. **Kernel density estimation.** A continuous density is estimated over $p \in [0,1]$ with a Gaussian kernel and Scott bandwidth: $\hat f(p) = \frac1{n}\sum_i K_h(p - p_i)$. The result is a smooth curve describing how the model's probability mass is distributed over contamination strength.
2. **Normalization.** Each density is divided by its own maximum so that its peak equals 1. This makes models with different sample counts comparable on *shape* rather than absolute frequency.
3. **Vertical stacking.** The nine normalized densities are placed along the vertical axis in a fixed order, separated by a constant offset, with the horizontal axis shared as $p \in [0,1]$. The vertical axis carries no numeric meaning; it only separates models.
4. **Annotation.** Each ridge is annotated with its mean (a dot plus the three-decimal value), and a vertical dashed line marks $\tau^*$.

**中译**：对每个模型汇总后的 $p$ 集合，执行以下步骤：

1. **核密度估计**：用高斯核与 Scott 带宽在 $p\in[0,1]$ 上估计连续密度 $\hat f(p)=\frac1n\sum_i K_h(p-p_i)$，得到一条平滑曲线，描述该模型的概率质量如何随污染强度分布。
2. **归一化**：每条密度除以自身最大值，使峰值等于 1。这使样本数不同的模型在"形状"上可比，而非绝对频数。
3. **纵向堆叠**：9 条归一化密度沿纵轴按固定顺序排列，相邻之间用常数间距隔开，横轴共用 $p\in[0,1]$。纵轴无数值含义，仅用于区分模型。
4. **标注**：每条山脊标注其均值（圆点 + 三位小数），并以竖直虚线标记 $\tau^*$。

## 1.2 图的目的是什么

**EN:** The purpose of the ridgeline plot is to move beyond the aggregate contamination *rate* (a single number per model) and expose the underlying **distribution of sample-level probabilities**. It asks: is contamination a binary on/off property, or a graded, continuous one? A rate of, say, 40% could arise from 40% of samples being fully contaminated and 60% fully clean, or from *all* samples being moderately contaminated—the rate alone cannot distinguish these, but the density can.

**中译**：山脊图的目的，是超越"每个模型一个污染率"的聚合数字，揭示底层**样本级概率的分布**。它要回答：污染究竟是二元的"有/无"，还是分级、连续的？例如 40% 的污染率，既可能来自"40% 样本完全污染、60% 完全干净"，也可能来自"所有样本都中度污染"——仅凭污染率无法区分二者，而密度曲线可以。

## 1.3 通过图能得到哪些结论

**EN:**

1. **Contamination is continuous, not binary.** The ridges span wide intervals of $p$ with substantial mass between $0$ and $1$, rather than collapsing onto the two endpoints. Contamination is therefore a matter of degree at the sample level.

2. **The negative/positive controls validate the detector.** CodeGen-6B-Multi concentrates at $p \approx 0$ (mean $\approx$ [0.01]) and Phi-2 at low $p$ (mean $\approx$ [0.09]), while RTLCoder concentrates at $p \approx 1$ (mean $\approx$ [0.97]). The detector assigns low scores to models known to be clean and high scores to a model known to be Verilog-fine-tuned, confirming it tracks genuine exposure.

3. **Intermediate models are heterogeneous mixtures.** Models between the two controls exhibit broad, often bimodal or long-tailed ridges straddling $\tau^*$, indicating clean, weakly contaminated, and heavily contaminated samples coexist within a single model.

4. **The aggregate rate hides sample-level structure.** Models with similar rates can have markedly different density shapes; the rate is thus a lossy summary.

5. **There is genuine uncertainty near the threshold.** The non-negligible mass around $\tau^*$ implies that the binary decision is unstable for a meaningful fraction of samples, so $\tau^*$ should not be treated as a hard boundary.

**中译：**

1. **污染是连续的，而非二元。** 山脊横跨较宽的 $p$ 区间，0 与 1 之间有可观质量，而非坍缩到两端。因此污染在样本级是一个程度问题。
2. **阴/阳对照校验了检测器。** CodeGen-6B-Multi 集中于 $p\approx0$（均值约 [0.01]）、Phi-2 集中于低 $p$（约 [0.09]），而 RTLCoder 集中于 $p\approx1$（约 [0.97]）。检测器对已知干净的模型给低分、对已知 Verilog 微调的模型给高分，证实它追踪的是真实暴露。
3. **中间模型是异质混合体。** 两对照之间的模型呈现宽、且常为双峰或长尾、横跨 $\tau^*$ 的山脊，表明同一模型内干净、弱污染、强污染样本并存。
4. **聚合率掩盖了样本级结构。** 污染率相近的模型其密度形状可能差异很大，污染率是有损的汇总。
5. **阈值附近存在真实不确定性。** $\tau^*$ 附近不可忽略的质量意味着，对相当一部分样本，二值决策不稳定，故不应把 $\tau^*$ 当作硬边界。

---

# 2. Lorenz 图（污染集中度偏差曲线）

**中译**：Lorenz 图（污染集中度偏差曲线）

## 2.1 数据是怎么算出来的

**EN:** For each model's pooled collection of $p$ values:

1. **Sort descending.** The $p$ values are sorted from largest to smallest, $p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(N)}$.
2. **Cumulative sample fraction.** The horizontal coordinate is $x_k = k/N$, the fraction of the $k$ most-contaminated samples.
3. **Cumulative contamination mass.** The vertical coordinate is $y_k = \sum_{i\le k} p_{(i)} / \sum_i p_{(i)}$, the fraction of total contamination mass (where mass is $\sum p$) carried by the top-$k$ samples.
4. **Deviation from uniformity.** Instead of the classical Lorenz curve $(x,y)$, we plot $z = y - x$, the *excess* mass share over the uniform baseline. If every sample contributed equal mass, $y=x$ and $z\equiv0$; if mass is concentrated, $z$ rises into a peak and falls back to zero.
5. **Gini coefficient.** $\mathrm{Gini} = 2\int_0^1 z\,dx$, twice the area under the deviation curve. It is $0$ for perfectly diffuse contamination and approaches $1$ for maximal concentration.
6. **Exclusion of near-clean models.** CodeGen-6B-Multi and Phi-2 are omitted because their total mass $\sum p \approx 0$, making the denominator vanish and the coefficient ill-defined.

**中译**：对每个模型汇总后的 $p$ 集合：

1. **降序排列**：将 $p$ 从大到小排列，$p_{(1)}\ge p_{(2)}\ge\dots\ge p_{(N)}$。
2. **累积样本比例**：横坐标 $x_k=k/N$，即最污染的 $k$ 个样本占总样本的比例。
3. **累积污染质量**：纵坐标 $y_k=\sum_{i\le k}p_{(i)}/\sum_i p_{(i)}$，即前 $k$ 个样本所承载的污染质量（质量定义为 $\sum p$）占比。
4. **偏离均匀**：不画经典洛伦兹曲线 $(x,y)$，而画 $z=y-x$，即超出均匀基线的质量占比。若每个样本贡献相同质量，$y=x$、$z\equiv0$；若质量集中，$z$ 先升成峰再回落到零。
5. **Gini 系数**：$\mathrm{Gini}=2\int_0^1 z\,dx$，即偏差曲线下面积的 2 倍。完全弥漫污染时取 0，最大集中时趋近 1。
6. **排除近干净模型**：CodeGen-6B-Multi 与 Phi-2 被省略，因其总质量 $\sum p\approx0$，分母趋于零、系数无定义。

## 2.2 图的目的是什么

**EN:** The purpose of the Lorenz-type curve is to answer a question the rate and the ridgeline plot cannot: **is contamination mass spread evenly across a benchmark, or concentrated in a small number of heavily memorized samples?** The same overall rate and even a similar mean $p$ can correspond to either regime, so a measure of *concentration* (inequality of mass over samples) is needed to tell them apart.

**中译**：Lorenz 型曲线的目的，是回答污染率与山脊图都无法回答的问题：**污染质量是均匀摊布在整个 benchmark，还是集中在少数被强记忆的样本上？** 相同的总体率、乃至相近的均值 $p$，都可能对应两种截然不同的情形，因此需要一个衡量"集中度"（质量在样本间的不平等程度）的指标来区分。

## 2.3 通过图能得到哪些结论

**EN:**

1. **Concentration differs sharply across models.** The seven substantially contaminated models span from near-uniform to markedly peaked curves, so *how* contamination is distributed is itself a distinguishing dimension.

2. **Diffuse exposure vs. localized memorization.** The most contaminated model, RTLCoder, lies near the uniform line (Gini $\approx$ [X]): nearly all its samples are contaminated, so the mass is necessarily spread—pervasive, benchmark-wide exposure. The moderately contaminated models ([模型名] Gini≈[X], [模型名] Gini≈[X]) rise sharply, meaning a minority of samples carries most of the mass—localized memorization of specific problems.

3. **Concentration adds information beyond the rate.** Two models with similar overall rates can differ sharply in concentration, so concentration is a complementary, not redundant, descriptor.

4. **It hints at training-data composition.** Diffuse exposure is consistent with whole-benchmark ingestion into training, whereas localized memorization is consistent with a few examples being over-represented. The two regimes suggest different contamination mechanisms.

**中译：**

1. **集中度在不同模型间差异显著。** 7 个有实质污染的模型从接近均匀到明显尖峰分布，说明污染"如何分布"本身就是一个有区分度的维度。
2. **弥漫暴露 vs 局部记忆。** 污染最重的 RTLCoder 贴近均匀线（Gini 约 [X]）：几乎全部样本都被污染，质量必然摊开——弥漫的、benchmark 级暴露。中等污染模型（[模型名] Gini≈[X]、[模型名] Gini≈[X]）曲线陡升，意味着少数样本承载了大部分质量——对特定题目的局部记忆。
3. **集中度为污染率之外提供增量信息。** 总体率相近的两个模型其集中度可能差别很大，故集中度是互补而非冗余的描述量。
4. **暗示训练数据构成。** 弥漫暴露与"整个 benchmark 被纳入训练"一致，局部记忆与"少数样本被过度重复"一致，两种模式暗示不同的污染机制。

---

# 3. 三个小节的分工（Progression of the three subsections）

**EN:** The three subsections are progressive: §3.1 answers *how much and where*; §3.2 descends to the sample level to show *how strong* contamination is as a continuum; §3.3 reveals *how concentrated* the mass is, separating diffuse exposure from localized memorization. Together they provide a complete characterization of benchmark contamination.

**中译**：三个小节递进：§3.1 回答"有多少、在哪里"；§3.2 下沉到样本级，展示污染作为连续谱"有多强"；§3.3 揭示污染质量"有多集中"，区分弥漫暴露与局部记忆。三者共同构成对 benchmark 污染的完整刻画。

---

# 附录：论文级描述（可直接精简进正文）

> 以下为三个小节的学术化英文正文与逐段中文翻译，可直接精简进论文对应小节。数值 `[ ]` 占位。

## 3.1 Overall Contamination Rates across Models and Benchmarks

**中译**：模型与 benchmark 层面的总体污染率

**EN（构建逻辑）：** To characterize contamination at the aggregate level, we apply the calibrated detector to every benchmark sample across all nine models and six benchmarks. Each sample is assigned a contamination probability $p$, and samples satisfying $p \ge \tau^*$—where $\tau^*$ denotes the OOF-optimized decision threshold—are classified as contaminated. The contamination rate of a (model, benchmark) pair is defined as the fraction of samples exceeding $\tau^*$, yielding a $9 \times 6$ rate matrix.

**中译**：为在聚合层面刻画污染，我们将已校准的检测器应用于全部 9 个模型、6 个 benchmark 的每一条样本。每条样本被赋予污染概率 $p$，凡满足 $p\ge\tau^*$（$\tau^*$ 为 OOF 优化的决策阈值）者被判定为污染。某 (模型, benchmark) 对的污染率定义为超过 $\tau^*$ 的样本占比，由此得到 $9\times6$ 的污染率矩阵。

**EN（结论）：** The resulting matrix reveals that contamination is a **model-specific property** rather than an intrinsic characteristic of the benchmarks. Although the same six benchmarks are evaluated across all models, contamination rates vary substantially with model. CodeGen-6B-Multi, a general-purpose model trained without Verilog-specific corpora, exhibits a near-zero rate ($\approx$ [0%]), and Phi-2 is only marginally contaminated ($\approx$ [9%]); both serve as negative controls. By contrast, RTLCoder—a model explicitly fine-tuned on Verilog—is almost entirely contaminated ($\approx$ [97%]), constituting a strong positive control. The remaining models ([DeepSeek-Coder $\approx$ X%], [Qwen2.5-Coder $\approx$ X%], [ChipGPT $\approx$ X%], [OriGen $\approx$ X%], [VeriCoder $\approx$ X%], [VeriGen $\approx$ X%]) occupy intermediate positions on a continuum. Because benchmark identity is held constant across models, this variation cannot be attributed to benchmark properties and instead reflects differences in the models' training data. The contrast between clean general-purpose models and heavily contaminated Verilog-specialized models further serves as internal evidence that the detector responds to genuine training-data exposure rather than to spurious statistical artifacts.

**中译**：该矩阵表明，污染是一种**模型特定的属性**，而非 benchmark 的内在特征。尽管同一批 6 个 benchmark 在所有模型上评估，污染率仍随模型显著变化。未用 Verilog 语料训练的通用模型 CodeGen-6B-Multi 呈现近零污染率（约 [0%]），Phi-2 仅轻微污染（约 [9%]），二者构成阴性对照；与之相对，明确用 Verilog 微调的 RTLCoder 几乎完全污染（约 [97%]），构成强阳性对照。其余模型（[DeepSeek-Coder ≈X%]、[Qwen2.5-Coder ≈X%]、[ChipGPT ≈X%]、[OriGen ≈X%]、[VeriCoder ≈X%]、[VeriGen ≈X%]）分布于两者之间的连续谱上。由于 benchmark 身份在各模型间保持不变，这一差异无法归因于 benchmark 属性，而只能反映各模型训练数据的差异。干净通用模型与重度污染的 Verilog 专用模型之间的反差，进一步构成内部证据，表明检测器响应的是真实训练数据暴露，而非虚假的统计伪影。

## 3.2 Contamination Intensity as a Continuous Spectrum

**中译**：作为连续谱系的污染强度

**EN（构建逻辑）：** To move beyond aggregate rates, we examine the full distribution of sample-level probabilities. For each model, the contamination probabilities $p$ of all samples across the six benchmarks are pooled, and a continuous density over $p \in [0,1]$ is estimated by kernel density estimation with a Gaussian kernel and Scott bandwidth. Each density is normalized by its maximum so that models are compared on distributional shape rather than raw sample counts, and the nine resulting densities are stacked vertically into a ridgeline plot. Each ridge is annotated with its mean (a dot with its numeric value), and a vertical dashed line marks the decision threshold $\tau^*$.

**中译**：为超越聚合率，我们考察样本级概率的完整分布。对每个模型，把其在 6 个 benchmark 上所有样本的污染概率 $p$ 汇总，并以高斯核、Scott 带宽的核密度估计，在 $p\in[0,1]$ 上估计连续密度。每条密度除以其最大值，使模型之间比较的是分布形状而非原始样本数，随后将 9 条密度纵向堆叠为山脊图。每条山脊标注其均值（圆点及数值），并以竖直虚线标记决策阈值 $\tau^*$。

**EN（结论）：** The ridgeline plot demonstrates that contamination is a **continuous spectrum rather than a binary state**. Had contamination been binary, each ridge would concentrate at $p \approx 0$ or $p \approx 1$; instead, the ridges span wide intervals of $p$ with non-negligible mass in between. The negative-control models are concentrated at low $p$ (CodeGen-6B-Multi, mean $\approx$ [0.01]; Phi-2, mean $\approx$ [0.09]), whereas the positive-control RTLCoder is concentrated near $p \approx 1$ (mean $\approx$ [0.97]). The intermediate models ([模型名] etc.) exhibit broad, frequently bimodal or long-tailed ridges straddling $\tau^*$, indicating a heterogeneous mixture of clean, weakly contaminated, and heavily contaminated samples. Two consequences follow. First, the aggregate rate reported in §3.1 constitutes only a coarse summary of an underlying continuum: models with comparable rates can nonetheless differ markedly in how their probability mass is distributed. Second, the substantial mass in the vicinity of $\tau^*$ indicates that the binary classification carries genuine uncertainty, cautioning against treating the threshold as a hard decision boundary.

**中译**：山脊图表明，污染是一个**连续谱系而非二元状态**。若污染为二元，每条山脊应集中于 $p\approx0$ 或 $p\approx1$；实际山脊却横跨较宽的 $p$ 区间，且区间内部存在不可忽略的质量。阴性对照模型集中于低 $p$（CodeGen-6B-Multi 均值约 [0.01]，Phi-2 约 [0.09]），而阳性对照 RTLCoder 集中于 $p\approx1$ 附近（均值约 [0.97]）。中间模型（[模型名] 等）呈现宽、且常为双峰或长尾、横跨 $\tau^*$ 的山脊，表明其样本是干净、弱污染与强污染样本的异质混合。由此产生两点推论。其一，§3.1 报告的聚合率只是底层连续分布的粗粒度汇总：污染率相近的模型，其概率质量的分布方式可能截然不同。其二，$\tau^*$ 附近可观的质量说明二值分类存在真实不确定性，提示不应将阈值视为硬性决策边界。

## 3.3 Concentration of Contamination

**中译**：污染集中度

**EN（构建逻辑）：** Finally, we investigate how contamination mass is distributed across samples, distinguishing **diffuse exposure** from **localized memorization**. For each model, samples are sorted by $p$ in descending order. We compute the cumulative sample fraction $x_k = k/N$ and the cumulative share of contamination mass $y_k = \sum_{i \le k} p_i / \sum_i p_i$, where contamination mass is defined as the total probability $\sum p$. Rather than the classical Lorenz curve $(x, y)$, we plot the deviation $z = y - x$, i.e., the excess contamination-mass share over the uniform baseline. Under perfectly uniform contamination, $z = 0$ identically; under concentrated contamination, $z$ rises into a peak before returning to zero. This deviation formulation amplifies the separation among curves, since the classical Lorenz curves all converge at $(0,0)$ and $(1,1)$ and thus overlap heavily. The Gini coefficient is defined as twice the area under the deviation curve, $\mathrm{Gini} = 2 \int_0^1 z\,dx$, taking the value $0$ for perfectly diffuse contamination and approaching $1$ for maximally concentrated contamination. Models with negligible total contamination mass (CodeGen-6B-Multi and Phi-2) are excluded, as the denominator $\sum p$ vanishes and the coefficient becomes ill-defined.

**中译**：最后，我们考察污染质量如何在样本间分布，以区分**弥漫暴露**与**局部记忆**。对每个模型，样本按 $p$ 降序排列。计算累积样本比例 $x_k=k/N$ 与累积污染质量占比 $y_k=\sum_{i\le k}p_i/\sum_i p_i$，其中污染质量定义为总概率 $\sum p$。我们不画经典洛伦兹曲线 $(x,y)$，而是画偏差 $z=y-x$，即超出均匀基线的污染质量占比。在完全均匀的污染下，$z\equiv0$；在集中污染下，$z$ 先升至峰值再回落到零。这一偏差形式放大了曲线间的分离度，因为经典洛伦兹曲线都在 $(0,0)$ 与 $(1,1)$ 处汇聚、彼此高度重叠。Gini 系数定义为偏差曲线下面积的 2 倍，$\mathrm{Gini}=2\int_0^1 z\,dx$，完全弥漫污染时取 0，最大集中污染时趋近 1。总污染质量可忽略的模型（CodeGen-6B-Multi 与 Phi-2）被排除，因为分母 $\sum p$ 趋于零，系数无定义。

**EN（结论）：** The seven substantially contaminated models separate into two distinct regimes. The most heavily contaminated model, RTLCoder, traces the uniform line $z = 0$ (Gini $\approx$ [X]): because nearly all of its samples are contaminated, the mass is necessarily spread across the benchmark, a signature of **pervasive, benchmark-wide exposure**. In contrast, the moderately contaminated models ([模型名], Gini $\approx$ [X]; [模型名], Gini $\approx$ [X]) exhibit markedly higher concentration, in which a minority of samples accounts for the bulk of the contamination mass—a signature of **localized memorization** of specific problems. This distinction refines the aggregate picture of §3.1: two models with similar overall rates may differ sharply in whether their contamination is a diffuse property of the model or the footprint of a small number of memorized examples. It further suggests differences in the underlying composition of training data—whole-benchmark ingestion versus the repetition of a few examples.

**中译**：7 个有实质污染的模型分为两种截然不同的模式。污染最重的 RTLCoder 贴近均匀线 $z=0$（Gini 约 [X]）：因其几乎所有样本都被污染，污染质量必然遍布整个 benchmark，这是**弥漫的、benchmark 级暴露**的特征。相反，中等污染模型（[模型名] Gini≈[X]、[模型名] Gini≈[X]）呈现显著更高的集中度，少数样本贡献了绝大部分污染质量，这是**对特定题目局部记忆**的特征。这一区分细化了 §3.1 的聚合图景：总体污染率相近的两个模型，在"污染是模型的弥漫属性，还是少数被记忆样本的足迹"上可能大相径庭。它进一步暗示了底层训练数据构成的差异——整库纳入与少数样本重复。

## 三节递进关系（Progression of the three subsections）

**EN:** The three subsections are organized progressively. §3.1 establishes *how much* contamination is present and *where* it occurs; §3.2 descends to the sample level to characterize *how strong* contamination is as a continuous spectrum; §3.3 reveals *how concentrated* the contamination mass is, separating diffuse exposure from localized memorization. Taken together, they provide a complete characterization of benchmark contamination.

**中译**：三个小节呈递进组织。§3.1 确立污染"有多少"且"在哪里"；§3.2 下沉到样本级，刻画污染作为连续谱系"有多强"；§3.3 揭示污染质量"有多集中"，区分弥漫暴露与局部记忆。三者共同构成对 benchmark 污染的完整刻画。
