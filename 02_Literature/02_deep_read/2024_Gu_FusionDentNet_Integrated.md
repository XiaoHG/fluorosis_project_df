---
date: 2026-05-21
author: LitAgent
input_from: "references/1-s2.0-S1746809424005688-main.pdf (BSPC 2024)"
output_to: "02_Literature/02_deep_read/2024_Gu_FusionDentNet_Integrated.md"
status: draft
---

# 精读摘要：Integrated Method for Grading Diagnosis of Dental Fluorosis Combined with Segmentation and Classification

## 文献信息

| 字段 | 内容 |
|------|------|
| 标题 | Integrated Method for Grading Diagnosis of Dental Fluorosis Combined with Segmentation and Classification |
| 作者 | Maohua Gu, Yun Wu*, Zhongchuan Jiang, Hao Xu |
| 年份 | 2024 |
| 出处 | Biomedical Signal Processing and Control, 96: 106510 |
| 机构 | 贵州大学 (State Key Lab of Public Big Data + College of CS) |
| 代码/数据 | https://github.com/yunwu2024/dental_fluorosis |

## 研究目标与任务类型

**任务**：氟斑牙分级诊断（二元分类：正常/病变；多分类：Normal/Mild/Moderate/Severe）。**这是深度学习应用于氟斑牙分级诊断的首次成功尝试**（截至2024年2月Web of Science检索无同类文献）。

核心挑战识别：(1) 氟斑牙病变区域形状不规则、边界模糊；(2) 轻度氟斑牙病变特征仅占图像极小比例；(3) 外源性染色、食物残渣、光照条件等干扰因素。

## 方法

**两阶段诊断框架**：

**Stage 1 — 牙齿区域分割**：
- 改进的U-Net：引入大核卷积残差模块 + SE注意力模块
- 大核卷积提取多尺度特征，增大有效感受野；SE模块聚焦感兴趣区域
- **像素关联迭代算法（PAI）**：后处理优化分割结果，考虑阈值设定和像素间关联性，在低阈值下保留牙齿区域的同时减少高阈值下的过分割
- 分割结果与原始图像融合后输入第二阶段

**Stage 2 — FusionDentNet分类器**：
- 双分支融合网络：CNN分支（ResNet50，局部细微病变特征）+ Transformer分支（LightViT，全局位置和粗糙度信息）
- **DBFM融合模块**：融合CNN局部特征和Transformer全局特征
- 注意力机制可解释性：模型关注区域随病变等级增加而扩大、变亮

## 关键结果

| 任务 | 指标 | 数值 |
|------|------|------|
| 二元分类（最佳配置） | Accuracy | 92.0%（正常）/ 100%（病变） |
| 多分类（FusionDentNet） | Accuracy | 80.0% |
| 多分类 | Precision | 0.813 |
| 多分类 | Sensitivity | 0.773 |
| 多分类 | mAP | 0.868 |

**消融实验关键发现**：
- 直接用原图分类效果差（ResNet50二元仅73.6%），验证了分割前置的必要性
- PAI后处理进一步提升准确率（LightViT多分类从72.7%→75.2%）
- DBFM特征融合显著优于单分支（FusionDentNet多分类80.0% vs ResNet50 70.0% vs DenseNet121 70.0%）

## 局限性

- 两阶段流程推理效率低（先分割再分类），不适合实时筛查
- 分割和分类模型独立训练，非端到端优化
- 数据集与MLTrMR相同的DFID，样本量有限（131例）
- 仅使用ResNet50+LightViT基础模型组合，未探索更新的backbone

## 可借鉴之处

1. **"先分割后分类"的两阶段策略**：已验证此范式对氟斑牙有效，本项目可沿用此思路——先提取牙齿ROI，再分级
2. **PAI后处理算法**：通用的分割优化方法，可迁移至本项目的数据预处理pipeline
3. **CNN+Transformer特征融合**：DBFM融合模块可作为本项目分类器的参考设计
4. **消融实验设计**：原文的消融矩阵（原图 vs 分割图 vs 分割+PAI + 单分支 vs 双分支）可直接复现为项目消融实验
5. **注意力可解释性可视化**：随病变等级增加关注区域变亮/扩大的发现为论文Discussion提供医学解读支撑

## 与本课题的关联点

**作为核心Baseline**：FusionDentNet是首个氟斑牙DL分级诊断方法（早于MLTrMR），本项目必须将其作为Baseline对比。两阶段范式与本项目可能的实验方案高度一致。该团队的数据集和代码完全开源，可直接复现其方法并在本项目200张数据集上评估。
