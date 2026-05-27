---
date: 2026-05-21
author: LitAgent
input_from: "references/Deep_Learning_Techniques_for_Dental_Image_Diagnostics_A_Survey.pdf"
output_to: "02_Literature/02_deep_read/2022_Salunke_Dental_DL_Survey.md"
status: draft
---

# 2022_Salunke_Dental_DL_Survey

**标题**：Deep Learning Techniques for Dental Image Diagnostics: A Survey

**作者**：Dipmala Salunke, Ram Joshi, Prasad Peddi, D. T. Mane (SJJTU Rajasthan & JSPM Pune, India)

**出处**：2022 International Conference on Augmented Intelligence and Sustainable Systems (ICAISS), IEEE. DOI: 10.1109/ICAISS55157.2022.10010576

---

## 研究目标与任务类型

对深度学习技术在牙科影像诊断中的应用进行**系统性综述**。涵盖龋齿检测、垂直根折检测、牙周病、牙齿检测/分类、牙菌斑检测等任务。

## 方法

- **检索策略**：PubMed, IEEE Xplore, arXiv.org, Journal of Oral Diseases, Dental Research, Periodontology
- **筛选结果**：200+ 篇文献 -> 44 篇纳入（2000-2021 年发表）
- **综述框架**：按应用领域分类（6 大类）+ 各方法的性能指标对比
- **涵盖架构**：U-Net, ResNet, VGG16, AlexNet, Faster R-CNN, Mask R-CNN, DetectNet, GoogLeNet Inception v3 等

## 关键结果与评价指标

- **龋齿检测**：MASK R-CNN 平均准确率 90%（2556 张）；GoogleNet Inception v3 准确率 89%（3000 张）；CNN+迁移学习 F1 0.75-0.83
- **氟斑牙**（提及但仅列出 1 项）：Liu 等用 MASK R-CNN 分类 1075 张氟斑牙图像
- **牙齿检测**：Faster R-CNN 灵敏度 0.9941，精确率 0.9945；VGG-16 性别分类 94.3%
- **牙周病**：CNN 准确率 81-98%
- **牙菌斑**：F1 0.75（CNN），MIoU 0.724（CNN）
- DL 方法普遍优于传统机器学习（SVM, K-NN）

## 局限性

1. 氟斑牙相关工作极度匮乏：综述仅找到 1 项直接相关工作（Liu 2015, 1075 张图像），说明该领域严重研究不足
2. 所有被综述的方法均基于 X 光/放射影像，而非口腔自然光照片——与本课题的模态不同
3. 综述质量一般：分类不够系统，缺乏对方法论的深度分析
4. 对氟斑牙分析的覆盖极其有限
5. 无代码开源汇总

## 可借鉴之处

1. **研究缺口确认**：综述明确指出氟斑牙 DL 分析"underexplored"，为本课题的创新性提供文献支撑
2. 各类牙科 DL 任务的性能基准可作为参考
3. 迁移学习策略（ImageNet 预训练 -> 牙科微调）在多个任务上验证有效
4. 数据增强在牙科 DL 中的普遍应用经验
5. 综述的参考文献列表可作为补充检索源

## 与本课题的关联点

- **高度相关**（⭐⭐⭐⭐⭐）：为本课题的文献综述章节提供直接支撑
- 确认了"氟斑牙+DL"领域的研究空白，是 Introduction 中论证创新性的关键文献
- 其牙科 DL 方法分类框架可借鉴到本课题的 Related Work 部分
- 对具体模型（ResNet, EfficientNet, ViT）在牙科任务上的表现总结可作为 baseline 选择的参考
