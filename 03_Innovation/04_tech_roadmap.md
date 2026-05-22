---
date: 2026-05-21
author: InnoAgent
input_from: "03_Innovation/03_selected_idea.md"
output_to: "03_Innovation/04_tech_roadmap.md"
status: draft
---

# 技术路线图：SymMamba + Ordinal-aware EDL 氟斑牙分级系统

## 1. 系统总览

```
                    输入: 512×256 口腔自然光照片
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  阶段 A: 口腔域 SSL 预训练 (离线, 一次性)                      │
  │  COde 50K + Oral Diseases 13K + AlphaDent 1.3K = ~64K 图像   │
  │  DINOv2 / MAE → 预训练权重                                   │
  └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  阶段 B: 氟斑牙域适应 (可选)                                  │
  │  200 张氟斑牙无标注图像 → Masked Image Modeling               │
  └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  阶段 C: SymMamba + Ordinal-aware EDL (核心)                 │
  │  ┌──────────────┐   ┌──────────────┐                         │
  │  │  Arch Scan   │   │  Cross Scan  │                         │
  │  │  (沿牙弓序列) │   │  (左右对称)   │                         │
  │  └──────┬───────┘   └──────┬───────┘                         │
  │         └────────┬─────────┘                                 │
  │                  ▼                                           │
  │         ┌───────────────┐                                    │
  │         │  CGF 交叉融合  │  (3 层级)                          │
  │         └───────┬───────┘                                    │
  │                 ▼                                            │
  │         ┌───────────────┐                                    │
  │         │   EDL 分类头   │ → Dirichlet α → u, b₀-b₃          │
  │         └───────────────┘                                    │
  │                                                              │
  │  L_total = L_EDL + L_ord + L_cont + L_boundary               │
  └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  阶段 D: 评估 + 诊断报告                                      │
  │  输出: Dean 分级 + 不确定性 + SDR + 结构化报告                │
  └─────────────────────────────────────────────────────────────┘
```

## 2. 阶段 A：口腔域自监督预训练

### 2.1 数据准备

| 数据集 | 图像数 | 模态 | 获取 |
|--------|--------|------|------|
| COde | ~50,000 | 口内 RGB | HuggingFace: zirak-ai/COde |
| Oral Diseases | ~13,000 | 口内 RGB | Kaggle: salmansajid05/oral-diseases |
| AlphaDent | ~1,320 | 口内 DSLR | GitHub |

合计 ~64,000 张口内自然光 RGB 图像，统一 resize 至 512×256，color normalization。

### 2.2 预训练方法

- **主方案**: DINOv2 (ViT-S/16), 100 epochs, AdamW, lr=1e-3
- **备选**: MAE (75% mask ratio)
- **硬件**: 1× A100 或 2× RTX 4090
- **质量验证**: 在 200 张氟斑牙数据上 linear probing，目标 Acc ≥ 55%

## 3. 阶段 B：氟斑牙域适应（可选）

仅在阶段 A linear probing Acc < 55% 时执行。

- 200 张氟斑牙图像（无标签）
- Masked Image Modeling: 60% mask, 50 epochs, lr=1e-4

## 4. 阶段 C：SymMamba + Ordinal-aware EDL

### 4.1 数据划分

- 5-fold CV，分层保持每类比例
- 每折: 160 train / 40 val

### 4.2 SymMamba 架构

```
512×256×3
  │
  ├─ Patch Embedding: Conv2d(k=4, s=4) → 128×64×D
  │
  ├─ Arch Scan: Mamba Block ×4 (沿行方向序列建模)
  │   └─ 输出 f_arch [128×64×D]
  │
  ├─ Cross Scan: Mamba Block ×4 (左右半区交叉扫描)
  │   └─ 输出 f_cross [128×64×D] + s_sym (对称性相似度)
  │
  ├─ CGF ×3: Gate ⊙ f_arch + (1-Gate) ⊙ f_cross
  │
  └─ GAP → z [D]
```

### 4.3 EDL 分类头

```
z [D] → Linear(D, 256) → ReLU → Linear(256, 4) → exp() + 1 → α [4]

S = Σα_k
b_k = (α_k - 1) / S    (信念质量)
u = 4 / S                (不确定性)
pred = argmax(α)         (预测类别)
```

### 4.4 损失函数

| 损失项 | 作用 | 权重 |
|--------|------|------|
| L_EDL | 标准证据损失 | 1.0 |
| L_ord | 有序正则化: 禁止信念质量跨类跳跃 | 0.1 |
| L_cont | 有序对比: 拉近相邻等级特征 | 0.05 |
| L_boundary | 边界不确定性先验: 类边界上 u 允许升高 | 0.01 |

### 4.5 训练配置

- AdamW, backbone lr=1e-4, EDL head lr=1e-3
- Cosine annealing, 5 epoch warmup
- Batch size 16, epochs 100 (early stop patience=20)
- RandAugment + CutMix + ColorJitter
- 5-fold CV 报告均值 ± std

### 4.6 不确定性阈值校准

在验证集上扫描 θ ∈ [0.1, 0.9]，选择使 SDR ≥ 0.95 下 Retention Rate 最大的 θ。

## 5. 阶段 D：评估与报告

### 5.1 评估指标

| 指标 | 用途 | 目标 |
|------|------|------|
| QWK | 与 MLTrMR 可比较 | ≥ 85% |
| SDR | 核心创新 | ≥ 95% |
| Macro F1 | 类均值 | ≥ 80% |
| ECE | 不确定性校准质量 | ≤ 0.10 |

### 5.2 消融实验矩阵

| 实验 | SymMamba | Arch | Cross | Ord EDL | SSL | 目的 |
|------|----------|------|-------|---------|-----|------|
| E1 | - | - | - | - | - | Baseline (ResNet50+CE) |
| E2 | - | - | - | ✓ | - | EDL 孤立效果 |
| E3 | ✓ | ✓ | - | - | - | Arch Scan only |
| E4 | ✓ | ✓ | ✓ | - | - | +Cross Scan, 无 EDL |
| E5 | ✓ | ✓ | ✓ | ✓ | - | 完整方案, 无 SSL |
| E6 | ✓ | ✓ | ✓ | ✓ | ✓ | **完整方案** |

### 5.3 对比方法

复现 MLTrMR / FusionDentNet / LD2Net / ResNet50+CORAL 在 5-fold CV 上的表现。

### 5.4 诊断报告

规则模板填充（不涉及 LLM），10 条规则覆盖分级-不确定性-对称性的全部组合。

## 6. 实施时间线

| 周次 | 阶段 | 任务 | 交付物 |
|------|------|------|--------|
| 第 5 周 | A | 下载预处理 64K 数据; DINOv2 预训练 | 预训练权重 |
| 第 6 周 | A+C | 预训练完成; 搭建 SymMamba 框架 | 代码框架 |
| 第 7 周 | C | 5-fold CV 训练; EDL 校准; 消融 | 训练日志, 权重 |
| 第 8 周 | C+D | 对比方法复现; 指标汇总; 报告模板 | 全部实验结果 |
| 第 9 周 | D | 结果分析; 可视化 | 图表初稿 |

## 7. 依赖

| 组件 | 来源 |
|------|------|
| mamba-ssm | PyPI |
| DINOv2 | facebookresearch (GitHub) |
| EDL head | 自实现 (~100 lines PyTorch) |
| 增强 | timm.data.RandAugment |
| 报告 | Python string.Template |
| 实验管理 | wandb |

## 8. 关键决策节点

| 节点 | 时间 | 决策 |
|------|------|------|
| D1 ✅ | 2026-05-21 | 选定创新方向 |
| D2 | 第 6 周末 | 确认 SymMamba 架构 (ModAgent) |
| D3 | 第 7 周末 | 审核实验方案 (ExpDesignAgent) |
| D4 | 第 8 周末 | 审阅结果，决定是否调整 |
| D5 | 第 9 周末 | 确认实验完成，进入图表制作 |
