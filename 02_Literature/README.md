---
date: 2026-05-21
author: LitAgent
input_from: "02_Literature/ 全部子目录产出, 00_Admin/project_plan.md, 01_Knowledge_Base/"
output_to: "02_Literature/README.md"
status: draft
---

# 02_Literature — 文献综述目录

## 目录内容索引

```
02_Literature/
├── README.md                              ← 本文件（目录说明与检索信息）
├── 01_initial_search/
│   └── literature_list.md                 # 初筛文献列表（30篇核心 + 16篇待评估）
├── 02_deep_read/                          # 精读摘要（共14篇）
│   ├── 1990_Fejerskov_Fluorosis_Mechanisms.md
│   ├── 1990_Cutress_Differential_Diagnosis_Fluorosis.md
│   ├── 2019_Dolz_HyperDenseNet.md
│   ├── 2021_Jain_KMeans_FCM_Fluorosis.md
│   ├── 2022_Petaitiemthong_EQIE_FCM_Seg.md
│   ├── 2022_Salunke_Dental_DL_Survey.md
│   ├── 2024_Ahmed_FluorosisCNN_Web.md
│   ├── 2024_Gu_FusionDentNet_Integrated.md
│   ├── 2024_Wu_MLTrMR_Fluorosis.md
│   ├── 2024_Zhou_CD_TransUNet.md
│   ├── 2025_Wu_HMDA.md
│   ├── 2025_Xu_MwincMamba_SkeletalFluorosis.md
│   ├── 2026_Li_LD2Net_Lightweight.md
│   └── 2026_Wu_CURSA_ViT.md
├── 03_comparison_table.md                 # 核心文献方法-数据-性能对比表
├── 04_gap_analysis.md                     # 研究缺口分析
└── 05_bibliography.bib                    # 参考文献库（BibTeX格式）
```

## 检索所用数据库

| 数据库/来源 | 用途 | 说明 |
|-------------|------|------|
| 人类用户预收集参考文献库 | 主要来源 | 162篇PDF，存放于 `/Volumes/KINGSTON/fluorosis_project/references/` |
| 文件名关键词筛选 | 初筛 | 基于PDF文件名进行关键词匹配和分类 |
| pdftotext 文本提取 | 精读 | 使用 `/opt/homebrew/bin/pdftotext` 从PDF提取全文文本 |
| arXiv | 预印本 | 参考文献库中包含多篇arXiv预印本（2404.13564v1等） |
| IEEE Xplore | 期刊/会议 | 主要IEEE期刊（TMI, JBHI, Access）和会议（INOCON, ICIRCA, ITC-CSCC, ICAISS, ICIPMC） |
| Elsevier/ScienceDirect | 期刊 | Biomedical Signal Processing and Control, Computers in Biology and Medicine, J Dent Res |
| Springer | 期刊/书籍 | Journal of Real-Time Image Processing, MICCAI proceedings |

> **说明**：由于参考文献库中绝大多数PDF为扫描图像（无嵌入文本层），Web搜索工具在本任务中不可用。实际检索策略为：对162篇PDF逐一进行文件名关键词分类 + pdftotext文本提取验证。此方法虽耗时但在当前条件下最可靠。

## 实际使用的全部检索式

### 文件名筛选关键词

| 检索式编号 | 关键词 | 用途 |
|-----------|--------|------|
| Q1 | `dental fluorosis`, `fluorosis`, `fluoride` | 氟斑牙/氟中毒直接相关 |
| Q2 | `deep learning`, `CNN`, `transformer`, `network`, `U-Net`, `Unet`, `ViT`, `SSM`, `Mamba` | DL方法 |
| Q3 | `segmentation`, `classification`, `grading`, `diagnosis`, `detection` | 任务类型 |
| Q4 | `medical`, `biomedical`, `dental`, `oral`, `X-ray` | 医学/牙科领域 |
| Q5 | `attention`, `fusion`, `multi-modal`, `ResNet`, `lightweight`, `masked` | 具体技术 |

### 等效数据库检索式

若在PubMed/Scopus等数据库中重建本检索，对应检索式如下：

```
# 检索式1：氟斑牙 + DL
("dental fluorosis" OR "enamel fluorosis") AND ("deep learning" OR "CNN" OR "transformer" OR "neural network")

# 检索式2：氟斑牙 + 分类/分级
("dental fluorosis" OR "enamel fluorosis") AND ("grading" OR "classification" OR "severity assessment" OR "automated diagnosis")

# 检索式3：牙科影像 + DL分割/分类
("dental" OR "oral") AND ("deep learning" OR "CNN") AND ("segmentation" OR "classification") AND ("image")

# 检索式4：氟中毒 + 自动化诊断
("fluorosis" OR "fluoride toxicity") AND ("automated diagnosis" OR "computer-aided" OR "deep learning")

# 检索式5：医学图像分割 SOTA（技术参考）
("medical image segmentation") AND ("transformer" OR "attention" OR "CNN") AND ("U-Net")
```

## 时间范围

- **主要范围**：2020–2026年（近5年）
- **经典文献**：1988–1990年（Fejerskov机制、Cutress鉴别诊断——氟斑牙病理与诊断的奠基性工作，不可替代）
- **传统方法**：2014–2023年（氟斑牙计算机辅助诊断的早期尝试，作为方法演进参照）

## 筛选原则与排序标准

### 优先级体系

| Tier | 范围 | 优先级 | 说明 |
|------|------|--------|------|
| Tier 1 | 氟斑牙DL直接应用 | ⭐⭐⭐⭐⭐ 最高 | 可直接作为Baseline或方法参考 |
| Tier 2 | 氟中毒临床/传统方法 | ⭐⭐⭐⭐ 高 | 临床背景支撑，方法演进参照 |
| Tier 3 | 牙科影像AI可迁移方法 | ⭐⭐⭐ 中 | 邻近任务的方法参照 |
| Tier 4 | 医学图像分割SOTA | ⭐⭐⭐ 中 | 技术组件可迁移至氟斑牙任务 |

### 纳入标准
1. 直接涉及氟斑牙/氟中毒的计算机辅助诊断或临床诊断 → Tier 1/2
2. 牙科影像的DL方法（分类/分割/检测），可迁移至氟斑牙 → Tier 3
3. 医学图像分割的SOTA方法，包含可迁移的技术组件（注意力机制、多模态融合、轻量化等） → Tier 4
4. 被引量高或发表于领域顶刊/顶会

### 排除标准
1. 氟骨症流行病学/治疗（保留方法学论文Mwinc-Mamba作为可迁移参考）
2. 环境/水质氟化物监测
3. 骨代谢/骨质疏松（非氟特异性）
4. 非医学图像分割（工业/遥感等）
5. 氟中毒相关的纯生化/分子生物学研究（与影像诊断无关）

## 文献统计

| 类别 | 数量 |
|------|------|
| 参考文献库总量 | 162篇PDF |
| 初筛纳入 | 30篇核心 + 16篇待评估 = 46篇 |
| 精读摘要 | 14篇 |
| 已排除 | ~116篇（氟骨症流行病学 ~10篇, 环境/水质 ~8篇, 骨代谢 ~5篇, 非医学分割 ~5篇, 其他不相关 ~88篇） |

### 精读文献覆盖分布

| 分组 | 篇数 | 代表文献 |
|------|------|----------|
| 氟斑牙DL方法 | 5 | MLTrMR, FusionDentNet, LD2Net, Ahmed 2024, Mwinc-Mamba(骨骼) |
| 氟斑牙传统方法 | 2 | EQIE-FCM, K-means+FCM |
| 临床/机制基础 | 3 | Fejerskov 1990, Cutress 1990, Wei 2019 |
| 医学图像分割SOTA | 4 | HyperDense-Net, CD-TransUNet, HMDA, CURSA |

## 后续步骤

1. 人类审核本目录全部产出（初筛列表、精读摘要、对比表、缺口分析、README、BibTeX）
2. 确认后，将 `02_Literature/` 移交至 **InnoAgent**（阶段2：创新点凝练）
3. InnoAgent 需重点阅读文件：`04_gap_analysis.md`（缺口）→ `03_comparison_table.md`（对比）→ 各精读摘要中的"可借鉴之处"和"与本课题的关联点"
