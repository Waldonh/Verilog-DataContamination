# 评估时的阈值逻辑：10 个单方法 vs 训练探测器

> 本文档详细说明方法比较实验中，10 个单方法与训练探测器（CatBoost-5f）在**评估阶段**各自采用何种阈值逻辑来计算 F1，以及为什么这样设计。

---

## 背景：F1 需要一个阈值，AUC 不需要

- **AUC** 是 ROC 曲线下面积，扫过所有可能的阈值，**与阈值无关**，是所有方法公平对比的主指标；
- **F1** 必须在某个阈值下先做"污染/干净"的二值判定，再算 precision 与 recall，因此**依赖阈值**。

所以"阈值怎么定"直接影响 F1，但不影响 AUC。下面分别说明两类方法的阈值逻辑。

---

## 1. 单方法的阈值逻辑：per-test-set 最优阈值（oracle）

10 个单方法（Zlib、Min-K%、Min-K%++、DC-PDD、PPL、PETAL、Neighbor、Reference、Token Overlap、CDD）是**规则方法**，它们的分数没有"训练阶段定好的阈值"（每个方法论文里给的固定阈值如 `zlib ≥ -0.003`，只是参考，未必是当前数据集上的最优）。

因此评估 F1 时，对每个单方法采用 **per-test-set 最优阈值**（oracle）：

- 在**每个测试集上单独**搜索一个使 F1 最大的阈值；
- 用该测试集自己的最优阈值计算 F1。

具体实现（遍历阈值网格取最大 F1）：

```python
def best_f1(y_true, prob):
    best_t, best = 0.5, -1.0
    for t in np.arange(0.05, 0.951, 0.005):        # 阈值网格 [0.05, 0.95], 步长 0.005
        f1 = f1_score(y_true, (prob >= t).astype(int))
        if f1 > best:
            best_t, best = float(t), f1
    return f1_score(y_true, (prob >= best_t).astype(int))   # 返回最优阈值下的 F1
```

**关键性质**：

- 这是 **oracle 阈值**——它"偷看"了测试集的真实标签来选最优阈值；
- **每个测试集独立**搜索（例如 HAVEN 测试集上的最优阈值，和 vericoder 测试集上的最优阈值可以不同）；
- 因此单方法的 F1 反映的是"该方法在最优阈值下的理论上限"。

---

## 2. 训练探测器的阈值逻辑：固定训练阈值 τ*

训练探测器（CatBoost-5f）在**训练阶段**就通过 out-of-fold 交叉验证确定了一个固定阈值 τ*：

- 训练集：CodeV + OriGen（C+O），1600 条（800 正 + 800 负）；
- 5 折分层交叉验证，得到每个训练样本的 OOF 预测概率；
- 在 OOF 预测上，遍历 `[0.05, 0.95]` 步长 `0.005`，取使 OOF F1 最大的阈值，即 **τ* = 0.4050**。

**评估阶段**，CatBoost-5f 的 F1 **直接使用这个固定 τ\***，不再针对测试集重新调：

```python
pred = (prob >= tau_star)                    # tau_star = 0.4050，固定
f1   = f1_score(y_test, pred)                # 用固定阈值算 F1
```

**关键性质**：

- 这是**固定阈值**——训练时一次定好，评估时不改、不偷看测试集；
- 因此 CatBoost-5f 的 F1 反映的是"**真实部署性能**"（部署时不可能偷看测试集标签，只能用训练时定好的阈值）。

---

## 3. 为什么这样设计：不对等的"更严格"对比

| | 单方法 | CatBoost-5f |
|---|---|---|
| 阈值 | per-test-set 最优（oracle，偷看测试集） | 固定训练阈值 τ*（不偷看） |
| 反映 | 方法的最佳可能 F1（理论上限） | 真实部署 F1 |
| 公平性 | 给了单方法有利的 oracle 上限 | 对自己更严格 |

这样设计的意义在于：**给单方法一个有利的 oracle 上限，同时让探测器用真实部署阈值**。如果在这种"不对等"条件下，CatBoost-5f 用固定 τ\* 的 F1 仍然优于单方法的 oracle F1，那么集成探测器的优势就是**真实、可部署**的，而不是靠偷看测试集阈值堆出来的。

以 6 测试集平均为例：

- CatBoost-5f（固定 τ\* = 0.405）F1 ≈ 0.8165；
- 最强单方法 DC-PDD（oracle）F1 ≈ 0.6613；

即使单方法用了 oracle、探测器用了固定阈值，探测器的 F1 仍高出约 0.155。

---

## 4. 论文表述建议

在方法对比处注明口径：

> "The single-method baselines are evaluated under per-test-set optimal thresholds (oracle), whereas our detector uses the fixed OOF-optimized threshold $\tau^*=0.405$ determined at training time. The detector's reported F1 therefore reflects its real deployment behavior, giving the baselines a favorable oracle upper bound. AUC is threshold-free and serves as the primary metric."

---

## 小结

- **单方法**：评估 F1 用 per-test-set 最优阈值（oracle），每个测试集独立搜最优阈值；
- **训练探测器**：评估 F1 用训练阶段 OOF 确定的固定阈值 τ* = 0.405；
- **AUC**：阈值无关，两方统一、是最公平的主指标；
- 这一"不对等但更严格"的设计，使探测器的 F1 优势更具说服力。
