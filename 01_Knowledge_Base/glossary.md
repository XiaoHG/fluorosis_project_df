---
date: 2026-05-14
author: Human
status: draft
---
# 统一术语表

本文档统一课题涉及的所有医学术语和深度学习术语的定义，供所有 Agent 在文献分析、创新点设计、模型开发、论文写作等环节使用，确保表述一致性。

---

## 1. 医学领域术语

### 1.1 疾病与病变

| 术语 | 英文 | 定义 |
|------|------|------|
| 氟中毒 | Fluorosis / Endemic Fluorosis | 因长期摄入过量氟化物引起的慢性全身性疾病 |
| 氟斑牙 | Dental Fluorosis | 釉质发育期过量摄氟导致的牙釉质结构异常 |
| 地方性氟中毒 | Endemic Fluorosis | 具有地区分布特征的氟中毒，与环境中高氟水平相关 |
| 白垩色病变 | Chalky White Opacity / White Spot Lesion | 牙釉质表面失去透明度的白色不透明区域，氟斑牙早期特征 |
| 釉质缺损 | Enamel Defect / Enamel Pitting | 釉质表面坑状或片状缺失，常见于重度氟斑牙 |

### 1.2 解剖部位

| 术语 | 英文 | 常用缩写 |
|------|------|----------|
| 切牙 | Incisor | — |
| 尖牙 | Canine | — |
| 前磨牙 | Premolar | — |
| 牙釉质 | Enamel | — |
| 牙本质 | Dentin | — |

### 1.3 诊断与分级

| 术语 | 英文 | 定义 |
|------|------|------|
| Dean 指数 | Dean's Fluorosis Index | 氟斑牙严重程度分级标准，本课题简化使用 0–3 级：0=正常，1=轻度，2=中度，3=重度 |
| 多分类分级 | Multi-class Grading | 氟斑牙严重程度分级：0=正常，1=轻度，2=中度，3=重度 |
| 分级一致性 | Inter-rater Agreement | 不同标注者间诊断结果的一致性，常用 Cohen's Kappa 评估 |
| 共识标签 | Consensus Label | 经专家讨论后达成的统一诊断标签 |

### 1.4 影像模态

| 术语 | 英文 | 说明 |
|------|------|------|
| 自然光照片 | White Light Image / Natural Light Photo | 标准白光下拍摄的口腔正面照片 |
| 荧光照片 | Fluorescence Image | 特定波长（如 405nm 蓝光）激发下的牙齿荧光成像，可显示早期脱矿区（注：当前数据集不包含此模态，属于未来扩展方向） |

### 1.5 流行病学

| 术语 | 英文 | 说明 |
|------|------|------|
| 高氟地区 | High-Fluoride Area | 饮水氟浓度 >1.5 mg/L 的地区 |
| 燃煤污染型氟中毒 | Coal-burning Type Fluorosis | 室内燃煤释放氟化物污染食物和空气所致，贵州为典型地区 |
| 饮水型氟中毒 | Drinking-water Type Fluorosis | 饮用高氟地下水所致 |

---

## 2. 深度学习领域术语

### 2.1 模型架构

| 术语 | 英文 | 说明 |
|------|------|------|
| 卷积神经网络 | Convolutional Neural Network (CNN) | 通过卷积核提取图像空间特征的深度学习架构 |
| 残差网络 | Residual Network (ResNet) | 引入残差连接解决深层网络退化问题，医学影像分析常用backbone |
| 视觉Transformer | Vision Transformer (ViT) | 基于自注意力机制的图像分类架构 |
| 双流网络 | Two-Stream Network / Dual-Stream Network | 分别处理不同模态输入并融合的网络架构 |
| 多模态融合 | Multi-modal Fusion | 融合不同成像模态信息的技术 |
| 注意力机制 | Attention Mechanism | 使模型聚焦输入中重要部分的机制，如SE、CBAM、自注意力 |
| 可解释性 | Explainability / Interpretability | 模型决策过程的可理解和可解释能力 |

### 2.2 融合策略

| 术语 | 英文 | 说明 |
|------|------|------|
| 早期融合 | Early Fusion / Input-level Fusion | 在输入层拼接多模态数据 |
| 中间融合 | Intermediate Fusion / Feature-level Fusion | 在特征提取中间层融合多模态特征 |
| 晚期融合 | Late Fusion / Decision-level Fusion | 各模态独立处理后融合决策结果 |
| 交叉注意力 | Cross-Attention | 一种模态的特征作为查询（Q），另一种模态作为键（K）和值（V）进行注意力计算 |
| 门控融合 | Gated Fusion | 使用门控机制自适应加权各模态特征 |

### 2.3 损失函数

| 术语 | 英文 | 说明 |
|------|------|------|
| 交叉熵损失 | Cross-Entropy Loss | 分类任务的标准损失函数 |
| 焦点损失 | Focal Loss | 降低易分类样本权重，缓解类别不平衡 |
| 排序损失 | Ranking Loss | 保持分级有序性的辅助损失 |
| 序数回归 | Ordinal Regression | 专为有序类别设计的回归框架 |

### 2.4 可视化技术

| 术语 | 英文 | 说明 |
|------|------|------|
| Grad-CAM | Gradient-weighted Class Activation Mapping | 基于梯度的类激活映射，可视化模型关注区域 |
| t-SNE | t-Distributed Stochastic Neighbor Embedding | 高维特征降维可视化方法 |
| 热力图 | Heatmap | 叠加在原始图像上的激活强度伪彩色图 |
| 混淆矩阵 | Confusion Matrix | 展示预测类别与真实类别对应关系的表格 |

### 2.5 评价指标

| 术语 | 英文 | 说明 |
|------|------|------|
| 准确率 | Accuracy (Acc) | 正确分类样本占总样本比例 |
| 精确率 | Precision | 预测为正例中真实正例的比例 |
| 召回率/敏感性 | Recall / Sensitivity | 真实正例中被正确检出的比例 |
| F1分数 | F1 Score | 精确率与召回率的调和平均 |
| Cohen's Kappa | Cohen's Kappa | 评估分类一致性的统计量，排除随机一致性影响，推荐作为分级任务的主指标 |
| 宏平均 | Macro Average | 各类别指标的算术平均，不受类别不平衡影响 |

### 2.6 训练策略

| 术语 | 英文 | 说明 |
|------|------|------|
| 迁移学习 | Transfer Learning | 将在源域预训练的模型迁移至目标域进行微调 |
| 数据增强 | Data Augmentation | 通过变换（旋转、翻转、颜色调整等）增加训练样本多样性 |
| 交叉验证 | Cross-Validation | 将数据集分为k份，轮流用k-1份训练、1份验证 |
| 早停 | Early Stopping | 验证指标不再提升时提前终止训练，防止过拟合 |
| 平衡采样 | Balanced Sampling / WeightedRandomSampler | 按类别权重采样，缓解不平衡 |
| 混合精度训练 | Mixed Precision Training | 同时使用 FP16 和 FP32 加速训练并节省显存 |

---

## 3. 课题特定缩写与约定

| 缩写 | 全称 | 使用场景 |
|------|------|----------|
| DL | Deep Learning | 深度学习，通篇可使用 |
| DF | Dental Fluorosis | 氟斑牙 |
| WL | White Light Image | 白光图像 |
| FL | Fluorescence Image | 荧光图像 |
| GT | Ground Truth | 真实标签 |
| MedIA | Medical Image Analysis | 目标期刊 |
| SOTA | State-of-the-Art | 当前最优方法 |
| CAD | Computer-Aided Diagnosis | 计算机辅助诊断 |

---

## 4. 术语使用规则

1. **全文一致性**：同一概念在论文中自始至终使用相同术语，首次出现时给出英文全称和缩写（如适用）。
2. **分级表述**：本课题的严重程度分级统一使用"0=正常，1=轻度，2=中度，3=重度"，不可与其他分级体系混淆。
3. **模态表述**：当前数据集为白光图像，统一称为"白光图像"或"自然光口腔照片"。荧光图像为未来扩展方向，在论文中提及时应明确标注为潜在方向。
4. **标注/标签**：指真实诊断结果（由口腔科医生提供）时使用"真实标签"或"GT标签"；指模型输出时使用"预测标签"。
5. **缩写管理**：首次在正文出现时标注全称，之后直接使用缩写；缩写表列入论文首稿后的术语页（如期刊要求）。