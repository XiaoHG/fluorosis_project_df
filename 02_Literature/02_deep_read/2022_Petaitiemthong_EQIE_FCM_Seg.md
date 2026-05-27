---
date: 2026-05-21
author: LitAgent
input_from: "references/Dental_Fluorosis_Segmentation_Using_Enhanced_Quantum-Inspired_Fuzzy_Clustering_Algorithm.pdf"
output_to: "02_Literature/02_deep_read/2022_Petaitiemthong_EQIE_FCM_Seg.md"
status: draft
---

# 2022_Petaitiemthong_EQIE_FCM_Seg

**标题**：Dental Fluorosis Segmentation Using Enhanced Quantum-Inspired Fuzzy Clustering Algorithm

**作者**：Natchapon Petaitiemthong, Sansanee Auephanwiriyakul, Nipon Theera-Umpon (Chiang Mai University), Chatpat Kongpun (ICOH, Thailand Ministry of Public Health)

**出处**：2022 37th International Technical Conference on Circuits/Systems, Computers and Communications (ITC-CSCC), IEEE. DOI: 10.1109/ITC-CSCC55581.2022.9894942

---

## 研究目标与任务类型

开发氟斑牙牙齿数字图像的**自动分割**算法，作为后续自动分类系统的前置步骤。任务类型为**语义分割**——将牙齿图像中的不同颜色区域（白色正常釉质、黄色轻中度病变、棕色重度病变）自动分离。研究背景：泰国清迈地区存在氟中毒问题，需自动化筛查减轻专家负担。

## 方法

- **颜色空间**：HSV（色调、饱和度、亮度），因 HSV 对颜色变化比 RGB 更敏感，能更好区分氟斑牙的白/黄/棕变色
- **核心算法**：Enhanced Quantum-Inspired Evolutionary Fuzzy C-Means (EQIE-FCM)
  - 基于量子比特编码模糊加权指数 m 和聚类中心
  - 使用量子旋转门在每代进化中更新量子比特
  - 采用 VIDSO 指数（综合簇内紧密度、簇间分离度、簇间重叠度）自动确定最优聚类数和模糊指数
- **多原型策略**：对白、黄、棕三种颜色分别运行 EQIE-FCM，生成各自的多原型聚类中心
- **最终分割**：K 近邻分类器（K=2）将每个像素分配到最近的颜色原型
- **数据集**：ICOH（泰国口腔健康国际中心）收集，训练集 7 张、验证集 16 张、盲测集 114 张

## 关键结果与评价指标

- 分割准确率：验证集 83.68%，盲测集 84.61%（与人工手动分割对比）
- EQIE-FCM 自动找到最优参数：白色 59 个原型（m=1.828）、黄色 57 个原型（m=1.934）、棕色 2 个原型（m=1.991）
- 算法参数：种群代数 gmax=100，旋转角 Delta_theta=0.03*pi，方差 sigma=0.6

## 局限性

1. 未实现最终的氟斑牙分类分级，仅为分割预处理步骤
2. 分割准确率约 84%，仍有提升空间，光照条件影响较大
3. 传统机器学习方法（模糊聚类+KNN），非深度学习方法，特征提取能力有限
4. 训练集极小（仅 7 张），泛化性存疑
5. 数据集来自泰国单一人群，对不同人种牙色的适应性未知

## 可借鉴之处

1. HSV 颜色空间的选用思路：氟斑牙的病变表现为颜色变化（白垩色、黄褐色），HSV 空间更易区分
2. 多原型策略：不同严重程度的病变颜色不同，分别建模更合理
3. 自动确定聚类数的 VIDSO 指标可参考用于无监督预分析
4. 数据采集流程值得参考（泰国 ICOH 的口腔筛查经验）
5. 团队后续工作（Wongkhuenkaew 2023, Fuzzy K-NN 分类）展示了完整 pipeline

## 与本课题的关联点

- **间接相关**（⭐⭐⭐⭐）：同为氟斑牙自动分析，但该工作仅做分割而非分级
- 其 HSV 颜色分析思路可迁移：本课题的 DL 模型可考虑 HSV 作为辅助输入通道
- EQIE-FCM 可作为无监督 baseline 或数据预标注工具
- 泰国团队的数据收集经验可参考用于本课题数据集扩展
