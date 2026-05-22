---
date: 2026-05-21
author: ExpDesignAgent
input_from: "02_Literature/03_comparison_table.md, 03_Innovation/03_selected_idea.md, 04_Model_Design/02_selected_architecture.md"
output_to: "05_Exp_Design/04_baselines.md"
status: draft
---

# 对比方法与复现方案

## 1. 对比方法选取原则

- **覆盖氟斑牙现有 SOTA**: MLTrMR (同级分类), FusionDentNet (检测+分类), LD2Net (轻量 CNN)
- **覆盖有序回归 baseline**: CORAL, Ordinal Regression CNN
- **覆盖不确定性 baseline**: 标准 EDL, Deep Ensemble, MC Dropout
- **覆盖 SSM baseline**: Mwinc-Mamba (骨骼氟中毒, 跨模态验证 SSM 有效性)
- 所有方法在同一 5-fold split 上评估, 确保可比性

## 2. Baseline 清单

### B1: ResNet50 (标准分类 baseline)

| 项目 | 规格 |
|------|------|
| Backbone | ResNet50 (ImageNet 预训练) |
| Head | GAP → Linear(2048, 4) |
| Loss | CrossEntropy |
| 训练 | AdamW, lr=1e-4, batch=16, epochs=100 |
| 输入 | 512×256, ImageNet 归一化 |
| 目的 | 确立性能下界 |
| 预期 QWK | ~75% |

### B2: ResNet50 + CORAL (有序回归 baseline)

| 项目 | 规格 |
|------|------|
| Backbone | ResNet50 (ImageNet 预训练) |
| Head | GAP → Linear(2048, 1) → K−1 个二分类器 |
| Loss | CORAL loss (Cao et al. 2020) |
| 训练 | AdamW, lr=1e-4, batch=16, epochs=100 |
| 目的 | 验证简单有序约束的效果 |
| 预期 QWK | ~78% |
| 参考 | Cao et al. "Deep Ordinal Regression with Label Diversity", NeurIPS 2020 |

### B3: MLTrMR (氟斑牙 SOTA, 必需复现)

| 项目 | 规格 |
|------|------|
| Backbone | ViT-B/16 (随机初始化, 原文设定) |
| Head | cls_token → MLP(768, 4) |
| Loss | CrossEntropy |
| 训练 | AdamW, lr=3e-4, batch=16, epochs=100 |
| 输入 | 224×224 (原文), 同时测试 512×256 |
| 目的 | 与当前氟斑牙最佳方法直接比较 |
| 预期 QWK | ~82% (原文报告) |
| 参考 | Wu et al. "MLTrMR: Multi-Level Transformer for Dental Fluorosis Grading", 2024 |

**复现注意**: 原文随机初始化 ViT (非 ImageNet 预训练), 本实验严格遵循。额外增加一个 ImageNet 预训练版本以做公平比较。

### B4: FusionDentNet (两阶段 baseline)

| 项目 | 规格 |
|------|------|
| Stage 1 | YOLOv8 牙齿检测 (如无检测标注, 跳过或使用全图) |
| Stage 2 | CNN+Transformer 混合编码器 → 4 类分类 |
| Loss | CrossEntropy |
| 训练 | 参照原文超参 |
| 目的 | 验证两阶段 vs 端到端的差异 |
| 预期 QWK | ~80% |
| 参考 | Gu et al. "FusionDentNet", 2024 |

**复现注意**: 本数据无牙齿检测标注。如无法获取检测模型, 跳过 Stage 1, 仅复现 Stage 2 分类器 (标注为 "FusionDentNet†")。

### B5: LD2Net (轻量 CNN baseline)

| 项目 | 规格 |
|------|------|
| Backbone | DSConv-based 自定义 CNN |
| Head | GAP → Linear |
| Loss | CrossEntropy |
| 训练 | Adam/SGD (参照原文) |
| 目的 | 验证轻量架构在小样本下的表现 |
| 预期 QWK | ~78% |
| 参考 | Li et al. "LD2Net: Lightweight Deep Dental Network", 2026 |

### B6: Mwinc-Mamba (SSM baseline, 跨域验证)

| 项目 | 规格 |
|------|------|
| Backbone | CNN + Mamba (单路径, 无 Cross Scan) |
| Head | Linear 分类头 |
| Loss | CrossEntropy |
| 目的 | 验证 Mamba/SSM 在氟斑牙照片上的有效性 (原文在骨骼 X 光) |
| 预期 QWK | ~79% |
| 参考 | Xu et al. "Mwinc-Mamba", 2025 |

**复现注意**: 原文数据为骨骼氟中毒 X 光 (SFXRay 80 张), 本实验为氟斑牙自然光。此 baseline 验证 "SSM 对氟中毒病变有效"的跨模态泛化性。

### B7: Deep Ensemble (不确定性 baseline)

| 项目 | 规格 |
|------|------|
| Backbone | ResNet50 × 5 |
| 不确定性 | 预测概率方差 (5 模型投票) |
| Loss | CrossEntropy |
| 目的 | 对比 EDL 单模型不确定性 vs Ensemble 不确定性 |
| 预期 QWK | ~80% (5 模型平均) |

### B8: MC Dropout (不确定性 baseline)

| 项目 | 规格 |
|------|------|
| Backbone | ResNet50 + Dropout(0.5) |
| 不确定性 | 30 次 MC forward pass 的预测熵 |
| Loss | CrossEntropy |
| 目的 | 对比 EDL 的 Dirichlet u vs MC Dropout 熵 |
| 预期 QWK | ~77% |

## 3. 公平比较规则

| 规则 | 措施 |
|------|------|
| 数据划分 | 所有方法使用同一 `split_indices.json` (5 折) |
| 增强 | Baseline 也使用同套增强管线 (公平对比 EDL 而非增强) |
| 输入尺寸 | 统一 512×256 (MLTrMR 额外测试 224×224) |
| 预训练 | 标注预训练来源 (ImageNet / 随机 / SSL) |
| 超参调优 | 每方法在各自验证集上调优最关键 2 个超参 |
| 指标 | 全部报告 QWK + Macro F1 + SDR(θ*) |

## 4. 对比矩阵

| 维度 | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 | **Ours** |
|------|----|----|----|----|----|----|----|----|------|
| 架构类别 | CNN | CNN | ViT | CNN+TF | CNN | CNN+SSM | Ensemble | MC | **SymMamba** |
| 有序建模 | ✗ | CORAL | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **Ord EDL** |
| 对称性 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **Cross Scan** |
| 不确定性 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Variance | Entropy | **Dirichlet u** |
| 预训练 | IN | IN | 随机 | IN | IN | 随机 | IN | IN | **Oral SSL** |
| 氟斑牙专有 | ✗ | ✗ | ✓ | ✓ | ✓ | 跨域 | ✗ | ✗ | **✓** |

## 5. 预期结果 (假设)

基于相似数据和文献, 预期 QWK 排序:

| 排名 | 方法 | 预期 QWK |
|------|------|----------|
| 1 | **Ours (E6, 完整)** | **≥ 87%** |
| 2 | Ours (E5, 无 SSL) | ~85% |
| 3 | MLTrMR + ImageNet | ~83% |
| 4 | MLTrMR (原文复现) | ~82% |
| 5 | FusionDentNet† | ~80% |
| 6 | Deep Ensemble (B7) | ~80% |
| 7 | Mwinc-Mamba (B6) | ~79% |
| 8 | ResNet50+CORAL (B2) | ~78% |
| 9 | LD2Net (B5) | ~78% |
| 10 | MC Dropout (B8) | ~77% |
| 11 | ResNet50 (B1) | ~75% |

**关键验证**: (1) Cross Scan 对称性 > 纯 SSM (B6); (2) Ordinal EDL > CORAL (B2); (3) Oral SSL > ImageNet (B3+IN vs Ours E6); (4) Dirichlet u > Ensemble/MC 不确定性。
