---
date: 2026-05-21
author: ExpDesignAgent
input_from: "01_Knowledge_Base/data_description.md, 04_Model_Design/02_selected_architecture.md, 03_Innovation/03_selected_idea.md"
output_to: "05_Exp_Design/03_metrics.md"
status: draft
---

# 评价指标体系

## 1. 指标总览

| 指标 | 全称 | 用途 | 优先级 |
|------|------|------|--------|
| QWK | Quadratic Weighted Kappa | 主指标, 与 baseline 可比 | ⭐⭐⭐ |
| SDR | Severe Detection Rate | 核心创新指标, 筛查安全性 | ⭐⭐⭐ |
| Macro F1 | Macro-averaged F1 | 类均值, 补充 | ⭐⭐ |
| ECE | Expected Calibration Error | 不确定性校准质量 | ⭐⭐ |
| UR-Acc | Uncertainty-Referred Accuracy | 部署效率参考 | ⭐ |
| Confusion Matrix | — | 定性分析 | ⭐ |

## 2. 主指标详述

### 2.1 QWK — Quadratic Weighted Kappa

**公式**:

```
κ = 1 - Σ w_ij · O_ij / Σ w_ij · E_ij
w_ij = (i - j)² / (K - 1)²
```

其中 O_ij 为观测混淆矩阵, E_ij 为期望混淆矩阵 (基于边际分布), w_ij 为二次权重矩阵。

**选择理由**:
- MLTrMR (Wu 2024) 报告 QWK, 本方案需可直接比较
- 二次加权惩罚远距离误判 (如 0→3) 重于邻近误判 (如 1→2), 符合 Dean 有序性
- MedIA 审稿人熟悉 QWK 作为有序分类指标

**目标值**: ≥ 85% (MLTrMR 报告 82%)

**计算**: `sklearn.metrics.cohen_kappa_score(y_true, y_pred, weights='quadratic')`

### 2.2 SDR — Severe Detection Rate

**公式**:

```
SDR(θ) = Recall(Severe | u ≤ θ)
       = TP_severe(θ) / (TP_severe(θ) + FN_severe(θ))
```

其中 θ 为不确定性阈值, 在验证集上校准。仅统计 u ≤ θ 的样本 (模型 confidence 足够高、无需转诊的子集)。

**选择理由**: 见创新点文档第 5 节。SDR 直接度量筛查安全性——"在系统自动判级的病例中, 重症是否全被抓出"。

**目标值**: SDR ≥ 95% (即 ≤5% 的重度患者被漏掉)

**校准流程**:
```
for θ in [0.1, 0.2, ..., 0.9]:
    confident_mask = (u ≤ θ)
    retention = mean(confident_mask)
    sdr = recall(Severe, confident_mask)
    if sdr ≥ 0.95: record(θ, retention, sdr)
select θ* with max retention among sdr ≥ 0.95
```

**报告**: 在论文中同时报告 SDR(θ*) 和对应的 Retention Rate。

## 3. 辅助指标

### 3.1 Macro F1

```
Macro F1 = (1/K) · Σ_k F1_k
F1_k = 2 · P_k · R_k / (P_k + R_k)
```

**用途**: 评估各类别均衡表现 (数据完全平衡, Macro F1 ≈ Accuracy, 但仍报告以透明)。

**目标值**: ≥ 80%

### 3.2 ECE — Expected Calibration Error

**公式**:

```
ECE = Σ_m (|B_m| / N) · |acc(B_m) - conf(B_m)|
```

将预测置信度分为 M=10 个 bin, 计算每个 bin 内准确率与平均置信度的差异。

**选择理由**: EDL 输出 Dirichlet α, 其中 u = K/Σα 作为预测不确定性。ECE 验证 u 是否被良好校准 (u 高时准确率确实低)。Deng (2024) 指出标准 EDL 常有校准问题, 需量化。

**目标值**: ECE ≤ 0.10

**计算**: `sklearn.calibration` 或自实现 (10 bins, 基于 max(b_k) 作为 confidence)。

### 3.3 UR-Acc — Uncertainty-Referred Accuracy

**公式**:

```
UR-Acc(θ) = Accuracy on {x: u ≤ θ}
```

**用途**: 度量"模型选择回答"的子集上准确率, 放入 Supplementary Material 作为 SDR 的补充。

**目标值**: UR-Acc(θ*) ≥ 85%

### 3.4 Class-wise Metrics

| 指标 | 公式 | 重点关注 |
|------|------|----------|
| Per-class Recall | TP_k / (TP_k + FN_k) | Severe 召回率 |
| Per-class Precision | TP_k / (TP_k + FP_k) | Normal 精确率 (减少假阳性转诊) |
| Per-class F1 | 2·P·R / (P+R) | Mild vs Moderate 边界类 |

## 4. 不确定性相关指标

### 4.1 Uncertainty Distribution

计算每类样本的平均 u 和标准差:

```
ū_k = mean(u_i | y_i = k)
σ_k = std(u_i | y_i = k)
```

**预期**: 边界类 (Mild, Moderate) 的 ū 应高于端类 (Normal, Severe), 验证 L_boundary 先验生效。

### 4.2 Rejection Curve

绘制 Retention Rate (u ≤ θ 的样本比例) vs QWK / SDR 曲线。x 轴=1−Retention (拒绝率), y 轴=指标。

**预期**: SDR 在低 rejection 时已接近 100%, QWK 随 rejection 上升而提升。

## 5. 对称性相关指标

### 5.1 s_sym 分布

统计 Cross Scan 对称性得分 s_sym ∈ [-1, 1] 的分布:
- 氟斑牙样本应偏向高 s_sym (>0.5)
- 如出现低 s_sym (<0) 集中, 可能为数据标注错误 (非对称性病变)

### 5.2 s_sym vs Uncertainty

绘制 s_sym 与 u 的散点图, 预期: 低对称性 + 高不确定性 → 模型判断为"不符合氟斑牙特征"。

## 6. 指标汇总报告模板

论文 Table 格式:

| Method | QWK (%) | SDR (%) | Macro F1 (%) | ECE |
|--------|---------|---------|--------------|-----|
| ResNet50 (baseline) | — | — | — | — |
| ResNet50 + CORAL | — | — | — | — |
| MLTrMR | — | — | — | — |
| FusionDentNet | — | — | — | — |
| LD2Net | — | — | — | — |
| **Ours (full)** | — | — | — | — |

所有指标报告 5-fold CV 均值 ± 标准差。

## 7. 统计检验

- **McNemar 检验**: Ours vs 每个 baseline (显著性 p<0.05)
- **Bootstrap CI**: 所有指标的 95% 置信区间 (1000 次 resample)
- **Fleiss' Kappa** (可选): 如有多标注者, 评估标注一致性
