---
date: 2026-05-21
author: LitAgent
input_from: "references/HMDA_A_Hybrid_Model_With_Multi-Scale_Deformable_Attention_for_Medical_Image_Segmentation.pdf"
output_to: "02_Literature/02_deep_read/2025_Wu_HMDA.md"
status: draft
---

# 2025_Wu_HMDA

**标题**：HMDA: A Hybrid Model With Multi-Scale Deformable Attention for Medical Image Segmentation

**作者**：Mengmeng Wu, Tiantian Liu, Xin Dai, Chuyang Ye, Jinglong Wu, Shintaro Funahashi, Tianyi Yan (Beijing Institute of Technology & Kyoto University)

**出处**：IEEE Journal of Biomedical and Health Informatics (JBHI), Vol. 29, No. 2, February 2025. DOI: 10.1109/JBHI.2024.3469230

---

## 研究目标与任务类型

提出 HMDA，一种融合 CNN 和 Transformer 的**混合医学图像分割**架构。核心目标：(1) 解决标准自注意力机制在高维数据上的计算冗余问题；(2) 实现 CNN 空间细节特征与 Transformer 长程上下文特征的有效显式交互。

## 方法

- **多尺度空间自适应可变形注意力 (MSADA)**：
  - 不计算所有 token 对的注意力，而是在多尺度特征中围绕参考点选择少量关键采样点进行计算
  - 显著降低计算复杂度，同时保持对关键区域的精确聚焦
- **交叉注意力桥 (CAB) 模块**：
  - 通过通道级交叉注意力实现多尺度 Transformer 特征与 CNN 局部特征的融合
  - 显式交互丰富特征合成
- **整体架构**：CNN encoder + Transformer encoder 并行 -> CAB 融合 -> Decoder
- **验证数据集**：多个医学图像分割数据集
- **资助来源**：国家重点研发计划、国家自然科学基金（多项）

## 关键结果与评价指标

- 在多个医学图像分割数据集上取得 competitive 性能
- MSADA 有效减少了计算冗余，同时保持甚至提升了分割精度
- CAB 模块显著提升了 CNN 和 Transformer 特征的融合质量
- 发表于 JBHI（IF≈7.0），方法学质量较高

## 局限性

1. 聚焦于分割任务，方法向分类/分级任务的直接迁移需要适配
2. 混合架构增加了工程实现复杂度
3. 多尺度可变形注意力的超参数（采样点数、尺度数）需要根据具体任务调优
4. 计算效率提升主要在长序列（大图像）场景下显著，对较小输入的优势不明显

## 可借鉴之处

1. **可变形注意力**：只关注关键采样点的思路可迁移到氟斑牙分类——模型应重点关注牙面白垩色病变区域而非整个图像
2. **CAB 交叉注意力融合**：可作为本课题特征融合模块的设计参考
3. CNN+Transformer 混合范式的最新实践（2025 年发表）
4. 多尺度特征处理：氟斑牙病变在不同尺度（细小白斑 vs 大面积白垩色）上表现不同，多尺度策略有直接应用价值

## 与本课题的关联点

- **方法参考**（⭐⭐⭐）：HMDA 的可变形注意力机制与本课题高度相关——氟斑牙分级任务天然需要模型聚焦于病变区域
- 若创新点涉及时病变区域注意力聚焦，HMDA 是必引的核心参考文献
- CAB 模块可作为 ModAgent 设计特征融合模块时的参考
- 发表于 2025 年，时效性强，有助于提升论文的文献前沿性
