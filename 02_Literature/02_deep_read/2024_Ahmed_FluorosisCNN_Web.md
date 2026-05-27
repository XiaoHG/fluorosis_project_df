---
date: 2026-05-21
author: LitAgent
input_from: "references/Dental_Fluorosis_Analysis_A_Web-Based_Dental_Fluorosis_Severity_Detection.pdf"
output_to: "02_Literature/02_deep_read/2024_Ahmed_FluorosisCNN_Web.md"
status: draft
---

# 2024_Ahmed_FluorosisCNN_Web

**标题**：Dental Fluorosis Analysis: A Web-Based Dental Fluorosis Severity Detection

**作者**：Sk. Zayeem Ahmed, K. Sri Kavya, R. Lalitha Karthik, S. Hemanth Mani (Velagapudi Ramakrishna Siddhartha Engineering College, India)

**出处**：2024 3rd International Conference for Innovation in Technology (INOCON), IEEE. DOI: 10.1109/INOCON60754.2024.10511698

---

## 研究目标与任务类型

开发基于深度学习的 Web 应用系统，用于氟斑牙严重程度自动检测和分类。任务为**多分类分级**结合**目标检测**（YOLO）。研究背景：印度 Vijayawada 市约 45% 居民有氟斑牙症状但大量未被诊断。

## 方法

- **模型架构**：CNN 负责严重程度分类 + YOLO 负责实时目标检测
- **数据集**：1182 张高质量牙齿图像，牙科专家标注严重程度等级
- **预处理**：标准化 + 数据增强
- **训练**：Categorical Cross-Entropy Loss + Adam 优化器，超参数调优
- **部署**：React 前端 + API Key 安全访问，支持实时摄像头检测

## 关键结果与评价指标

- 分类准确率：96%（严重程度分类）
- 目标检测精确率：84%
- 声称较前驱工作（Wongkhuenkaew 2023, Fuzzy K-NN）提升 13.33%

## 局限性

1. 数据集来源单一（单一合作者），缺乏多中心验证
2. 分级标准不明确，未说明使用 Dean 还是 TF 指数
3. 缺乏与 SOTA 方法的系统对比，仅与一项前驱工作比较
4. 评价指标不完整：仅报告 Accuracy 和 Precision，缺 F1、Kappa、混淆矩阵
5. 会议论文（4 页），技术深度有限，无代码开源

## 可借鉴之处

1. CNN+YOLO 组合思路：先检测牙面区域再分类
2. Web 实时检测 + 移动端兼容的部署模式
3. 数据集规模参考：1182 张标注图像
4. 参考文献网络涵盖氟斑牙机制和牙科 DL 综述关键文献

## 与本课题的关联点

- **直接相关**（⭐⭐⭐⭐⭐）：同为氟斑牙严重程度自动分级
- 可作为对比 baseline，但其性能报告（96%）在仅 200 张训练集下可能过于乐观
- Web 部署设计可供未来 Demo 参考
- 标注流程（牙科专家标注）与本课题类似
