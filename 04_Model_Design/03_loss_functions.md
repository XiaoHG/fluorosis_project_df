---
date: 2026-05-21
author: ModAgent
input_from: "03_Innovation/03_selected_idea.md, 04_Model_Design/01_architecture_variants/"
output_to: "04_Model_Design/03_loss_functions.md"
status: draft
---

# 损失函数汇总

## 总损失

```
L_total = L_EDL + λ1·L_ord + λ2·L_cont + λ3·L_boundary [+ λ4·L_sym (仅 V2)]
```

---

## 1. L_EDL — 证据深度学习主损失

```
L_EDL(α, y) = Σ_k (y_k - b_k)² + Σ_k y_k · (S - α_k)² / S

α_k = e_k + 1          浓度参数 (e_k ≥ 0)
S   = Σ α_k            Dirichlet 强度
b_k = (α_k - 1) / S    信念质量
y_k ∈ {0,1}            one-hot 标签
```

第一项最小化预测信念与标签的 MSE，第二项压制非目标类的证据。优势：输出不确定性 u = K/S，在数据噪声大或类边界模糊时自动升高。氟斑牙标注噪声大（牙医一致性 <70%），EDL 天然建模标注不确定性。

参考: Sensoy M, Kaplan L, Kandemir M. "Evidential Deep Learning to Quantify Classification Uncertainty." NeurIPS 2018.

---

## 2. L_ord — 有序信念正则化

```
L_ord = Σ_{k=1}^{K-2} max(0, b_{k-1} + b_{k+1} - 2·b_k)
```

惩罚信念质量 b_k 在类别间跳跃。Dean 0→1→2→3 是有序病理过程，证据分布应平滑过渡。**λ1 = 0.1**，调参范围 [0.05, 0.2]。

---

## 3. L_cont — 有序对比损失

```
L_cont = -log[ Σ_{j:|y_j-y_i|≤1} exp(cos(z_i,z_j)/τ) /
               Σ_{all j} exp(cos(z_i,z_j)/τ) ]
```

拉近相邻等级样本的特征 (|y_j-y_i|≤1)，推远跨级样本。在表示学习层面编码有序性。**λ2 = 0.05, τ = 0.07**。

---

## 4. L_boundary — 边界不确定性先验

```
L_boundary = - Σ_{j:|y_j-y_i|=1} log(u_i)
```

当 batch 内有邻级样本时，鼓励当前样本的不确定性适度升高。使类边界（Mild/Moderate交界）的 u 偏高 → 标记为转诊 → 减少硬分类错误。**λ3 = 0.01**。

---

## 5. L_sym — 对称性辅助损失（仅 Variant 2）

```
L_sym = BCE(s_sym, y_sym_label)
```

s_sym ∈ [0,1] 为对称性得分。伪标签在训练 5 epoch 后由 SFE 差异统计值通过 K-means(k=2) 自动生成。**λ4 = 0.05**。

---

## 权重汇总与调参

| 损失项 | 默认权重 | 调参范围 | 优先级 |
|--------|---------|---------|--------|
| L_EDL | 1.0 | 固定 | 核心 |
| L_ord | 0.1 | 0.05–0.2 | 高 |
| L_cont | 0.05 | 0.02–0.1 | 中 |
| L_boundary | 0.01 | 0.005–0.02 | 低 |
| L_sym (V2) | 0.05 | 0.02–0.1 | 低 |

调参建议：
- 若训练初期 L_ord 过大导致所有 b_k 趋近均匀（u 始终很高），减小 λ1
- 若 L_cont 导致特征坍缩（所有 z 相似），增大 τ 或减小 λ2
- 若 u 在所有样本上接近 1（过度不确定），减小 λ3；接近 0（过度自信），增大 λ3
