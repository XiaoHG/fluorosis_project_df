---
date: 2026-05-21
author: LitAgent
input_from: "references/Detection_of_dental_Fluorosis_using_enhanced_K-means_and_Fuzzy_C_means.pdf"
output_to: "02_Literature/02_deep_read/2021_Jain_KMeans_FCM_Fluorosis.md"
status: draft
---

# 2021_Jain_KMeans_FCM_Fluorosis

**标题**：Detection of Dental Fluorosis Using Enhanced K-Means and Fuzzy C-Means

**作者**：Richa Jain, Devesh Saini, Aakash Singal, Rishabh Tiwari (Amity University & GTBIT, India)

**出处**：2021 Third International Conference on Inventive Research in Computing Applications (ICIRCA), IEEE. DOI: 10.1109/ICIRCA51532.2021.9544836

---

## 研究目标与任务类型

提出一种**无人工干预**的氟斑牙严重程度检测方法，通过无监督聚类算法自动计算氟斑牙病变面积百分比，并将图像划分为四级严重程度。任务为**无监督图像分割 + 基于面积的分级**。

## 方法

- **颜色空间转换**：RGB -> HSV，提取 HS（色调+饱和度）图像。理由：氟斑牙病变区域呈黄色，与正常白色釉质在 HSV 空间区分度更高
- **ROI 提取**：手动或自动提取牙齿区域作为感兴趣区域
- **聚类算法**：
  - Enhanced K-Means（K=2）：将 HS 图像像素分为正常区和病变区
  - Fuzzy C-Means（FCM）：作为 K-Means 的交叉验证方法
- **严重程度分级**：计算病变面积百分比，按阈值分级
  - Very Mild: 0-10%
  - Mild: 10-20%
  - Moderate: 20-30%
  - Severe: >30%
- **数据集**：来自印度 AIIMS Delhi 的牙齿图像，由牙科专家标注严重程度作为 ground truth

## 关键结果与评价指标

- 10 张测试图像中，9 张的分级结果与专家标注一致
- K-Means 与 FCM 两种方法的结果高度一致（面积百分比差异 <1%）
- 验证了 HSV 颜色空间对氟斑牙病变检测的有效性
- 严重程度分级基于客观的面积百分比阈值，减少了主观性

## 局限性

1. 测试集极小（仅 10 张图像），无法评估统计显著性
2. 仅依赖颜色信息（HS 图像），忽略了纹理和形态学特征
3. K=2 的硬聚类假设（正常/病变二分）过于简化：氟斑牙表现为连续谱而非二值
4. 面积百分比阈值（10%/20%/30%）与 Dean 指数的对应关系未严格验证
5. 传统聚类方法，非深度学习，无法端到端学习特征表示
6. 牙齿 ROI 提取方法未详细说明，实际应用中可能引入误差

## 可借鉴之处

1. 氟斑牙病变面积百分比作为定量指标的思路，可融入 DL 方法的辅助输出
2. HSV 颜色空间的有效性验证：可考虑在 DL 模型输入中增加 HSV 通道
3. 无监督方法在小样本场景下的应用价值：本课题 200 张数据量偏小，可结合无监督预训练
4. 与专家的一致性对比方法：9/10 的对比实验设计可参考

## 与本课题的关联点

- **间接相关**（⭐⭐⭐⭐）：同为氟斑牙分级，但使用传统方法
- 可作为无监督 baseline 对比方法
- HSV 空间 + 面积百分比的思路可融入 DL pipeline 作为辅助特征
- 严重程度与面积百分比的定量关系值得在本课题数据上验证
