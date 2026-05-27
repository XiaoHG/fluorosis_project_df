---
date: 2026-05-21
author: LitAgent
input_from: "references/CD-TransUNet_Enhanced_UNet_for_Medical_Image_Segmentation_via_Global_and_Local_Feature_Fusion.pdf"
output_to: "02_Literature/02_deep_read/2024_Zhou_CD_TransUNet.md"
status: draft
---

# 2024_Zhou_CD_TransUNet

**标题**：CD-TransUNet: Enhanced UNet for Medical Image Segmentation via Global and Local Feature Fusion

**作者**：Junkai Zhou, Ben Ye (Macau University of Science and Technology)

**出处**：2024 3rd International Conference on Image Processing and Media Computing (ICIPMC), IEEE. DOI: 10.1109/ICIPMC62364.2024.10586680

---

## 研究目标与任务类型

提出 CD-TransUNet，一种融合 CNN 卷积块和 Transformer 注意力机制的医学图像**分割**框架。核心目标：解决 UNet 变体在局部特征提取和全局特征提取之间的平衡问题，增强特征保留和梯度流动。

## 方法

- **架构设计**：Encoder-Decoder 结构（UNet 范式）
  - Encoder：CNN 卷积块提取局部特征 + Transformer 注意力块捕获全局长程依赖
  - 通道拼接（Channel Concatenation）：在特征聚合中引入额外的跳跃连接，增强特征保留并缓解梯度消失
  - Decoder：上采样恢复空间分辨率
- **核心创新**：在 UNet 的跳跃连接中引入通道维度拼接，使解码器能同时利用编码器的多尺度特征
- **训练配置**：[从上下文推断] 标准医学图像分割训练策略

## 关键结果与评价指标

- 在医学图像分割任务上表现优于标准 UNet 和注意力 UNet 变体
- 通道拼接策略有效缓解了深层网络训练中的梯度问题
- 全局+局部特征融合提升了分割精度，特别是在边界模糊区域

## 局限性

1. 会议论文，篇幅有限（4 页），技术细节不够充分
2. 未报告具体的 Dice 分数和 IoU 值（从提取的文本中未找到）
3. 缺乏在多种医学影像模态（CT/MRI/X-ray/照片）上的广泛验证
4. 计算复杂度增加：Transformer 模块引入了额外参数量
5. 无代码开源信息

## 可借鉴之处

1. CNN+Transformer 混合架构设计思路：本课题的氟斑牙分级模型可借鉴此混合范式，CNN 提取局部纹理特征（釉质细节），Transformer 捕获全局牙齿形态
2. 通道拼接特征保留策略：可迁移到本课题的多尺度特征融合设计
3. UNet 结构在医学影像中的成熟性：即使本课题是分类任务，UNet 的 encoder 部分可作为 backbone
4. 局部-全局特征平衡的方法论：氟斑牙分级既需要关注局部病变纹理（局部特征），也需要整体牙面评估（全局特征）

## 与本课题的关联点

- **方法参考**（⭐⭐⭐）：为模型架构设计提供 CNN+Transformer 混合思路
- 其通道拼接策略可启发本课题的特征融合模块设计
- 若本课题创新点涉及注意力机制或多尺度特征（如 InnoAgent 示例 idea_1），CD-TransUNet 是重要参考
- 可作为 Related Work 中 UNet 系列方法演进的代表文献
