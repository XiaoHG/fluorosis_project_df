---
date: 2026-05-21
author: LitAgent
input_from: "01_Knowledge_Base/ 全部文件, /Volumes/KINGSTON/fluorosis_project/references/ (162 PDFs)"
output_to: "02_Literature/01_initial_search/literature_list.md"
status: draft
---

# 文献初筛列表

## 检索概况

- **检索来源**：人类用户预收集的参考文献库（162 篇 PDF）+ 文件名关键词筛选
- **筛选原则**：优先牙科氟中毒直接应用 > 牙科影像可迁移方法 > 医学图像分割 SOTA 方法 > 临床背景文献
- **初筛数量**：从 162 篇中筛选 30 篇核心文献
- **时间范围**：以 2020-2025 年为主，含少量经典文献
- **去重状态**：文件名去重已完成，DOI 去重待补充

---

## 初筛文献列表

### A. 氟斑牙直接应用（Tier 1 — 最高优先级）

| # | 标题 | 第一作者 | 年份 | 出处 | 方法/任务 | 相关性 |
|---|------|----------|------|------|-----------|--------|
| 1 | Dental Fluorosis Analysis: A Web-Based Dental Fluorosis Severity Detection | [待确认] | 2024 | [待确认] | 氟斑牙严重程度检测，Web 应用，DL 分类 | ⭐⭐⭐⭐⭐ |
| 2 | Dental Fluorosis Segmentation Using Enhanced Quantum-Inspired Fuzzy Clustering Algorithm | [待确认] | 2023 | [待确认] | 量子启发模糊聚类，氟斑牙分割 | ⭐⭐⭐⭐⭐ |
| 3 | Detection of Dental Fluorosis Using Enhanced K-means and Fuzzy C-means | [待确认] | 2022 | [待确认] | 增强 K-means + FCM，氟斑牙检测 | ⭐⭐⭐⭐⭐ |
| 4 | Deep Learning Techniques for Dental Image Diagnostics: A Survey | [待确认] | 2023 | [待确认] | 牙科影像 DL 综述，含分类/分割/检测 | ⭐⭐⭐⭐⭐ |
| 5 | Semantic Segmentation on Panoramic Dental X-Ray Images Using U-Net Architectures | Zannah R | 2024 | IEEE Access | 全景牙科 X 光 U-Net 语义分割 | ⭐⭐⭐⭐ |
| 5a | Masked Latent Transformer with Random Masking Ratio to Advance the Diagnosis of Dental Fluorosis | Wu Y, Xu H | 2024 | arXiv:2404.13564v1 | MLTrMR, ViT掩码潜在建模，DFID数据集，SOTA | ⭐⭐⭐⭐⭐ |
| 5b | Integrated Method for Grading Diagnosis of Dental Fluorosis Combined with Segmentation and Classification | Gu M, Wu Y | 2024 | Biomed Signal Process Control 96:106510 | 两阶段法：大核U-Net分割+FusionDentNet双分支分类 | ⭐⭐⭐⭐⭐ |
| 5c | Convolutional State Space Model with Multi-Window Cross-Scan to Advance the Automated Diagnosis of Skeletal Fluorosis | Xu H, Wu Y | 2025 | Biomed Signal Process Control 103:107439 | Mwinc-Mamba，CNN+SSM双分支，骨骼氟中毒（方法可迁移） | ⭐⭐⭐⭐ |
| 5d | LD2Net: Real-Time Lightweight Fluorosis Grading with Depthwise Separable Convolution and Dual-Axis Attentional Intelligence | Li M, Wu Y | 2026 | J Real-Time Image Proc 23:85 | 轻量级（3.31M参数），迁移学习，三维注意力 | ⭐⭐⭐⭐⭐ |

### B. 氟斑牙临床与机制（Tier 2 — 临床背景支撑）

| # | 标题 | 第一作者 | 年份 | 出处 | 方法/任务 | 相关性 |
|---|------|----------|------|------|-----------|--------|
| 6 | Differential Diagnosis of Dental Fluorosis | [待确认] | 2022 | [待确认] | 氟斑牙鉴别诊断，临床特征分析 | ⭐⭐⭐⭐ |
| 7 | The Nature and Mechanisms of Dental Fluorosis in Man | Fejerskov O | 1988 | J Dent Res | 氟斑牙病理机制经典文献 | ⭐⭐⭐⭐ |
| 8 | The Pathogenesis of Endemic Fluorosis: Research Progress in the Last 5 Years | Wei W | 2019 | J Cell Mol Med | 氟中毒发病机制综述 | ⭐⭐⭐ |
| 9 | Fluoride, Teeth and Bone | Smith GE | 1985 | Med J Aust | 氟化物对牙齿和骨骼影响的经典综述 | ⭐⭐⭐ |
| 10 | IJEM-21-190 — [氟中毒流行病学相关] | [待确认] | [待确认] | Indian J Endocr Metab | 地方性氟中毒流行病学 | ⭐⭐⭐ |

### C. 牙科影像 AI 可迁移方法（Tier 3 — 方法参考）

| # | 标题 | 第一作者 | 年份 | 出处 | 方法/任务 | 相关性 |
|---|------|----------|------|------|-----------|--------|
| 11 | Artificial Intelligence and Other Modern Digital Technologies in Dentistry | [待确认] | 2023 | [待确认] | 牙科 AI/数字技术综述 | ⭐⭐⭐⭐ |
| 12 | Int J Comm Dent 2023 — [口腔公共卫生相关] | [待确认] | 2023 | Int J Comm Dent | 口腔公共卫生，氟斑牙流行病学 | ⭐⭐⭐ |
| 13 | Sustainability-15-12227 — [氟中毒与环境健康] | [待确认] | 2023 | Sustainability | 氟中毒环境健康影响 | ⭐⭐ |

### D. 医学图像分割 SOTA 方法（Tier 4 — 技术参考，可迁移）

| # | 标题 | 第一作者 | 年份 | 出处 | 方法/任务 | 相关性 |
|---|------|----------|------|------|-----------|--------|
| 14 | CD-TransUNet: Enhanced UNet for Medical Image Segmentation via Global and Local Feature Fusion | [待确认] | 2023 | [待确认] | Transformer+CNN 混合分割 | ⭐⭐⭐ |
| 15 | HMDA: A Hybrid Model With Multi-Scale Deformable Attention for Medical Image Segmentation | [待确认] | 2025 | [待确认] | 多尺度可变形注意力混合模型 | ⭐⭐⭐ |
| 16 | MaS-TransUNet: A Multi-Attention Swin Transformer U-Net for Medical Image Segmentation | [待确认] | 2024 | [待确认] | 多注意力 Swin Transformer UNet | ⭐⭐⭐ |
| 17 | DPCF-Net: 2D Medical Image Segmentation Network Based on Dual-Path Cross-Fusion Encoder | [待确认] | 2024 | [待确认] | 双路径交叉融合编码器 | ⭐⭐⭐ |
| 18 | MCBTNet: Multi-Feature Fusion CNN and Bi-Level Routing Attention Transformer | [待确认] | 2024 | [待确认] | CNN+Transformer 多特征融合 | ⭐⭐⭐ |
| 19 | LogTrans: Providing Efficient Local-Global Fusion with Transformer and CNN Parallel Network | [待确认] | 2024 | [待确认] | Transformer+CNN 并行局部-全局融合 | ⭐⭐⭐ |
| 20 | HyperDense-Net: A Hyper-Densely Connected CNN for Multi-Modal Image Segmentation | [待确认] | 2019 | IEEE TMI | 超密集连接 CNN，多模态分割 | ⭐⭐⭐ |
| 21 | REDNet: Reliable Evidential Discounting Network for Multi-Modality Medical Image Segmentation | [待确认] | 2024 | [待确认] | 证据理论学习，多模态分割 | ⭐⭐⭐ |
| 22 | EviVLM: When Evidential Learning Meets Vision Language Model for Medical Image Segmentation | [待确认] | 2024 | [待确认] | 证据学习+VLM 分割 | ⭐⭐ |
| 23 | Asymmetric Adaptive Heterogeneous Network for Multi-Modality Medical Image Segmentation | [待确认] | 2023 | [待确认] | 多模态非对称自适应异构网络 | ⭐⭐ |
| 24 | CPA-UNet: nnUNet-Based Local Pyramid Aggregation Network | [待确认] | 2023 | [待确认] | 局部金字塔聚合 UNet | ⭐⭐ |
| 25 | U-Shiftformer: Brain Tumor Segmentation Using A Shifted Attention Mechanism | [待确认] | 2023 | [待确认] | 移位注意力机制分割 | ⭐⭐ |
| 26 | META-Unet: Multi-Scale Efficient Transformer Attention Unet for Polyp Segmentation | [待确认] | 2024 | [待确认] | 多尺度高效 Transformer UNet | ⭐⭐ |
| 27 | TransDeep: Transformer-Integrated DeepLabV3 for Image Semantic Segmentation | [待确认] | 2025 | [待确认] | Transformer+DeepLabV3 语义分割 | ⭐⭐ |
| 28 | The CUR Decomposition of Self-Attention Matrices in Vision Transformers | [待确认] | 2024 | [待确认] | ViT 自注意力矩阵 CUR 分解，轻量化 | ⭐⭐⭐ |
| 29 | A Few-Shot Learning Framework for the Diagnosis of Osteopenia and Osteoporosis Using Knee X-Ray Images | Xie et al. | 2024 | [待确认] | 小样本学习+骨密度诊断，可迁移至小样本氟斑牙 | ⭐⭐⭐ |
| 30 | Prior Information Guided Coarse-to-Fine Dual-Branch Encoding Network for Fovea Localization and Optic Disc Cup Segmentation | [待确认] | 2023 | [待确认] | 先验引导双分支分割，粗到细策略 | ⭐⭐ |

---

## 补充文献（参考文献库中待进一步评估）

以下文献存在于用户提供的参考文献库中，相关性需精读阶段进一步确认：

| # | 文件名 | 推测内容 | 潜在价值 |
|---|--------|----------|----------|
| 31 | `1-s2.0-S1746809423008911-main.pdf` | Elsevier 论文，可能为牙科/医学影像 DL | 待读 |
| 32 | `1-s2.0-S1746809424014976-main.pdf` | 2024 年医学影像相关 | 待读 |
| 33 | `1-s2.0-S0010482524000222-main.pdf` | Computers in Biology and Medicine | 待读 |
| 34 | `1-s2.0-S089360802400577X-main.pdf` | Neural Networks 期刊 | 待读 |
| 35 | `1-s2.0-S0950705123007372-main.pdf` | Knowledge-Based Systems | 待读 |
| 36 | `1-s2.0-S1568494624010299-main.pdf` | Applied Soft Computing | 待读 |
| 37 | `2404.13564v1.pdf` (31 MB) | arXiv 2024 大文件，可能为综述或博士论文 | 待读 |
| 38 | `2407.08083v2.pdf` | arXiv 2024 | 待读 |
| 39 | `2411.19331v3.pdf` (45 MB) | arXiv 2024 超大文件，可能为综述 | 待读 |
| 40 | `2505.22195v2.pdf` (19 MB) | arXiv 2025 | 待读 |
| 41 | `978-3-030-87193-2.pdf` (~128 MB) | Springer 书籍，可能为 MICCAI proceeding 或医学影像专著 | 待读 |
| 42 | `978-3-030-87196-3.pdf` (~104 MB) | Springer 书籍 | 待读 |
| 43 | `978-3-031-72120-5.pdf` (~115 MB) | Springer 书籍 | 待读 |
| 44 | `1-s2.0-S1746809424005688-main.pdf` | 2024 医学影像相关 | 待读 |
| 45 | `1-s2.0-S1746809424008978-main.pdf` | 2024 医学影像相关 | 待读 |
| 46 | `1-s2.0-S1746809424012126-main.pdf` | 2024 医学影像相关 | 待读 |

---

## 排除文献（明确不相关或出界）

以下类别文献存在于参考文献库中，经评估与课题（氟斑牙 DL 分级）相关性低，已排除：

- **氟骨症相关**（已排除约 10 篇）：`Association+of+Dietary+Carotenoids+Intake+with+Skeletal+Fluorosis*.pdf`, `Integrated_Learning_Approach_Based_on_Fused_Segmentation_Information_for_Skeletal_Fluorosis*.pdf`, `Treatment+and+Prevention+of+Skeletal+Fluorosis.pdf`, `neurology_of_endemic_skeletal_fluorosis.4.pdf`, `moving_on_from__no_trace__wrong_place__for_the.1.pdf`, `s00198-014-2707-4.pdf`, `s11914-021-00701-y.pdf` 等
- **环境/水质氟化物**（已排除约 8 篇）：`1-s2.0-S0048969706008345-main.pdf`, `s10653-021-01148-x.pdf`, `s10661-022-10888-x.pdf` 等
- **骨代谢/骨质疏松**（已排除约 5 篇）：`s40572-020-00270-9.pdf`, `s40572-023-00412-9.pdf` 等
- **非医学图像分割**（已排除约 5 篇）：`Improved_SwinUnet-based_image_segmentation_method_for_charcoal_slag*.pdf`, `remotesensing-17-00802.pdf`, `A_Token-Sparsity-Based_Image_Panoptic_Segmentation_Method_for_Natural_Scenes.pdf` 等

> **总计**：收录 30 篇核心 + 16 篇待评估 = 46 篇纳入后续考虑。已排除约 116 篇。

---

## 检索式与关键词

用于文件名筛选和分类的关键词组合：

| 检索式编号 | 关键词 | 用途 |
|-----------|--------|------|
| Q1 | `dental fluorosis`, `fluorosis`, `fluoride` | 氟斑牙直接相关 |
| Q2 | `deep learning`, `CNN`, `transformer`, `network`, `U-Net`, `Unet` | DL 方法 |
| Q3 | `segmentation`, `classification`, `grading`, `diagnosis`, `detection` | 任务类型 |
| Q4 | `medical`, `biomedical`, `dental`, `oral` | 医学/牙科领域 |
| Q5 | `attention`, `fusion`, `multi-modal`, `ViT`, `ResNet` | 具体技术 |

---

## 下一步

1. 请人类审核初筛列表，确认文献覆盖度和相关性
2. 审核通过后，进入 **精读阶段**：从本列表中选取 Top 10 文献逐篇深度摘要
3. 补充文献（待评估 16 篇）需人工确认是否纳入精读
4. `[待确认]` 标记的元数据需通过 PDF 阅读或 DOI 查询补充
