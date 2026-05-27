---
date: 2026-05-21
author: LitAgent
input_from: "references/1-s2.0-S1746809424014976-main.pdf (BSPC 2025)"
output_to: "02_Literature/02_deep_read/2025_Xu_MwincMamba_SkeletalFluorosis.md"
status: draft
---

# 精读摘要：Convolutional State Space Model with Multi-Window Cross-Scan to Advance the Automated Diagnosis of Skeletal Fluorosis

## 文献信息

| 字段 | 内容 |
|------|------|
| 标题 | Convolutional State Space Model with Multi-Window Cross-Scan to Advance the Automated Diagnosis of Skeletal Fluorosis |
| 作者 | Hao Xu, Yun Wu*, Rui Xie, Jun Xu, Junpeng Wu, Rongpin Wang, Youliang Tian |
| 年份 | 2025 |
| 出处 | Biomedical Signal Processing and Control, 103: 107439 |
| 机构 | 贵州大学, 织金县人民医院, 南京信息工程大学, 贵州省人民医院 |
| 代码/数据 | https://github.com/uxhao-o/Mwinc-Mamba |

## 研究目标与任务类型

**任务**：骨骼氟中毒X光片自动分级诊断（二元分类：正常/异常；多分类：Normal/Mild/Moderate/Severe）。

注意：本文研究**骨骼氟中毒**（skeletal fluorosis）而非氟斑牙（dental fluorosis），但两者属于同一课题组（贵州大学Wu团队）的系列工作，方法高度可迁移。骨骼氟中毒是氟斑牙的严重进展形式，两者共享病因（过量氟摄入），在地方性氟中毒筛查中常需同时诊断。

## 方法

**数据集 — SFXRay**：
- 全球首个开源骨骼氟中毒X光数据集
- 122例原始X光片，筛选后80张入组（前臂/骨盆/小腿三个部位）
- 3位10年以上经验的放射科专家独立标注，多数投票确定标签
- 21例正常，59例病变（34 Mild + 13 Moderate + 12 Severe），按7:3划分训练/测试集

**模型架构 — Mwinc-Mamba**：
- **双分支结构**：CNN分支（局部特征提取）+ SSM分支（长程依赖建模）
- CNN弥补SSM不擅长提取局部特征的短板
- **Multi-Window Cross-Scan机制**（核心创新）：将patches划分至多个窗口并执行交叉扫描，捕获多粒度病变特征
- 基于Mamba（状态空间模型），线性复杂度，适合高分辨率X光图像

## 关键结果

| 任务 | 指标 | 数值 |
|------|------|------|
| 二元分类（正常/异常） | Accuracy | 83.33% |
| 多分类（4级） | Accuracy | 66.67% |
| 多分类 vs 放射科医生均值 | 差距 | 仅3.33% |

- 多分类准确率与5位受邀放射科医生平均水平无显著差异，临床潜力强
- 复杂度显著低于Transformer-based方法
- 5位放射科医生间Cohen's kappa一致性差（热力图显示），凸显自动化诊断需求

## 局限性

- 数据量极小（80张X光，最小类仅12张），严重限制模型性能
- 骨骼氟中毒病灶特征不明显，多分类66.67%仍远低于临床可接受水平
- 仅包含前臂/骨盆/小腿X光，未涵盖其他受累部位（脊柱、关节等）
- 标注依赖专家投票，仍存在标注噪声

## 可借鉴之处

1. **SSM/Mamba架构应用于氟中毒影像**：Mamba的线性复杂度可迁移至氟斑牙任务，特别适合高分辨率口腔图像
2. **Multi-Window Cross-Scan**：多窗口交叉扫描策略可改造用于氟斑牙牙面多区域特征捕获
3. **CNN+SSM双分支融合范式**：本项目中可设计CNN（局部纹理）+ SSM（牙面全局结构）的混合模型
4. **放射科医生间一致性作为参照基线**：论文的医生间诊断一致性分析方法可直接应用于氟斑牙诊断的临床动机论述
5. **公开数据集+代码**：可作为迁移学习源域或多任务学习的辅助任务

## 与本课题的关联点

**方法迁移参考**：虽然任务不同（骨骼 vs 牙齿），但同为氟中毒影像分级，同属贵州大学Wu团队系列工作，方法栈高度一致。Mwinc-Mamba的SSM双分支架构可改造为氟斑牙分级模型。该团队的DFID + SFXRay构成氟中毒多器官影像体系，未来多任务联合学习有探索空间。在本项目Related Work中应作为氟中毒DL诊断的姊妹工作引用。
