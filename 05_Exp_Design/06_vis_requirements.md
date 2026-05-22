---
date: 2026-05-21
author: ExpDesignAgent
input_from: "04_Model_Design/02_selected_architecture.md, 03_Innovation/03_selected_idea.md, 05_Exp_Design/03_metrics.md, 05_Exp_Design/05_ablation_plan.md"
output_to: "05_Exp_Design/06_vis_requirements.md"
status: draft
---

# 可视化需求清单

## 1. 图表规划总览

靶向 MedIA 投稿: 正文 6-8 张图 + 3-4 张表, Supplementary Material 不限。

| 编号 | 类型 | 标题 | 位置 | 优先级 |
|------|------|------|------|--------|
| Fig 1 | 示意图 | 系统架构总览 | 正文 | ⭐⭐⭐ |
| Fig 2 | 示意图 | SymMamba 双路径详解 + Cross Scan | 正文 | ⭐⭐⭐ |
| Fig 3 | 数据图 | 混淆矩阵 (Ours vs MLTrMR) | 正文 | ⭐⭐⭐ |
| Fig 4 | 数据图 | SDR 曲线 + Rejection Curve | 正文 | ⭐⭐⭐ |
| Fig 5 | 数据图 | 不确定性分析 (ū per class, s_sym vs u) | 正文 | ⭐⭐ |
| Fig 6 | 热力图 | Grad-CAM 对比 (Ours vs baseline) | 正文 | ⭐⭐ |
| Fig 7 | 示意图 | 诊断报告示例 | 正文 | ⭐⭐ |
| Fig 8 | 数据图 | t-SNE 特征分布 (E0 vs E4 vs E6) | 正文 | ⭐⭐ |
| Tab 1 | 表格 | Baseline 对比总表 | 正文 | ⭐⭐⭐ |
| Tab 2 | 表格 | 消融实验总表 | 正文 | ⭐⭐⭐ |
| Tab 3 | 表格 | Per-class metrics | 正文/Supp | ⭐⭐ |
| Tab 4 | 表格 | 损失函数消融 (L0-L3) | Supp | ⭐ |
| Fig S1 | 数据图 | CGF Gate 可视化 (3 stages) | Supp | ⭐ |
| Fig S2 | 数据图 | SSL Linear Probing 结果 | Supp | ⭐ |
| Fig S3 | 表格 | 不确定性阈值校准详情 | Supp | ⭐ |

---

## 2. Fig 1: 系统架构总览

**类型**: 示意图 (矢量, SVG/PDF)

**内容**:
- 输入: 512×256 口内照片
- 4 阶段流程: SSL 预训练 (可选) → SymMamba 编码 → EDL 分类 → 诊断报告输出
- 标注各组件名称和功能
- 底部标注: "Dean 0→1→2→3 有序分级"

**尺寸**: 单栏 (90mm 宽) 或双栏 (190mm), 根据复杂度选择
**分辨率**: 矢量 PDF, ≥300 dpi raster 部分

---

## 3. Fig 2: SymMamba 双路径详解

**类型**: 结构示意图 (矢量)

**内容**:
- 左侧: Arch Scan — 128×64 特征图 → 沿行方向展开为序列 → Mamba Block ×4
- 右侧: Cross Scan — 左右半区 split → 各自 Mamba → Concat → 对称性得分计算
- 中部: CGF ×3 门控融合
- 标注: 通道数 (96→192→384), 尺寸缩减 (128×64→64×32→32×16)
- 右下角: s_sym 计算公式

**尺寸**: 双栏 (190mm)
**分辨率**: 矢量 PDF

---

## 4. Fig 3: 混淆矩阵对比

**类型**: 数据图 (Matplotlib/Seaborn heatmap)

**内容**:
- 两个并排混淆矩阵: Ours (E6) vs MLTrMR (B3)
- 4×4 矩阵, 归一化至行和=1 (recall)
- 颜色: 对角线深绿, 非对角线浅红
- 标注: 每格百分比 + 样本数

**数据来源**: 5-fold CV 汇总 (全部 200 样本, 每样本在作为 val 时的预测)

**尺寸**: 双栏 (190mm)
**分辨率**: 300 dpi PNG + PDF

---

## 5. Fig 4: SDR 曲线 + Rejection Curve

**类型**: 双轴数据图

**内容**:
- 左 y 轴: SDR (0-1, 蓝色实线)
- 右 y 轴: Retention Rate (0-1, 红色虚线)
- x 轴: θ ∈ [0.1, 0.9]
- 垂直线: θ* (SDR≥0.95 处最大 retention)
- 底部: 标注 θ* 值和对应 SDR/Retention

**数据来源**: 验证集扫描

**尺寸**: 单栏 (90mm)
**分辨率**: 300 dpi PNG + PDF

---

## 6. Fig 5: 不确定性分析

**子图 A**: ū per class (bar chart)
- x 轴: Dean 0/1/2/3
- y 轴: mean u ± std
- 预期: Mild/Moderate ū 高于 Normal/Severe (验证 L_boundary)

**子图 B**: s_sym vs u 散点图
- x 轴: s_sym (对称性得分)
- y 轴: u (不确定性)
- 颜色: Dean label (4 色)
- 预期: 低 s_sym + 高 u 区域聚集边界样本

**尺寸**: 双栏 (190mm, 两个子图水平排列)
**分辨率**: 300 dpi PNG + PDF

---

## 7. Fig 6: Grad-CAM 对比

**类型**: 热力图叠加在原始图像上

**内容**: 3 行 × 4 列网格
- 行: Ours (E6) / MLTrMR (B3) / ResNet50 (B1)
- 列: Dean 0/1/2/3 各一例
- 每格: 原始图像 + 热力图叠加 (jet colormap, alpha=0.5)
- 标注: 真实标签 + 预测标签 + 置信度/证据量

**预期差异**: Ours 的热力图应集中在釉质病变区域 (白垩色/着色区域), 且双侧区域激活对称。

**尺寸**: 双栏 (190mm)
**分辨率**: 300 dpi PNG + PDF (原始图像不压缩)

---

## 8. Fig 7: 诊断报告示例

**类型**: 文字模板渲染 (非 AI 生成图)

**内容**: 2 个示例面板
- 左侧: 正确分级 (u 低, s_sym 高, Dean 2 → Moderate)
- 右侧: 不确定转诊 (u 高, s_sym 中等, Dean 1 → Mild, 建议转诊)
- 格式: 等宽字体, 模拟终端输出

**尺寸**: 单栏 (90mm)
**格式**: 矢量 PDF

---

## 9. Fig 8: t-SNE 特征分布

**类型**: 散点图

**内容**: 3 个面板 (水平排列)
- E0 (ResNet50+CE): 预期类别混杂
- E4 (SymMamba+EDL, 无 SSL): 预期类别分离, Dean 0/1/2/3 大致有序排列
- E6 (完整方案): 预期最佳分离, 特征沿 Dean 序数轴分布

**数据**: GAP 后的特征向量 z ∈ R^768, t-SNE 降至 2D, perplexity=30

**尺寸**: 双栏 (190mm)
**分辨率**: 300 dpi PNG + PDF

---

## 10. 表格需求

### Tab 1: Baseline 对比总表

| Method | QWK (%) | SDR (%) | Macro F1 (%) | ECE | Params |
|--------|---------|---------|--------------|-----|--------|
| (B1-B8 + Ours E6) | mean±std | mean±std | mean±std | mean±std | M |

附带统计显著性标记 († p<0.05 vs Ours, McNemar).

### Tab 2: 消融实验总表

| ID | Arch Scan | Cross Scan | Ord EDL | SSL | QWK (%) | SDR (%) |
|----|-----------|------------|---------|-----|---------|---------|
| E0 | | | | | | |
| E1 | | | ✓ | | | |
| ... | | | | | | |
| E6 | ✓ | ✓ | ✓ | ✓ | | |

### Tab 3: Per-class Metrics (Ours E6)

| Class | Precision | Recall | F1 | Mean u |
|-------|-----------|--------|-----|--------|
| Normal (0) | | | | |
| Mild (1) | | | | |
| Moderate (2) | | | | |
| Severe (3) | | | | |

### Tab 4: 损失函数消融 (Supplementary)

| Loss | QWK | SDR | ECE |
|------|-----|-----|-----|
| L0: L_EDL only | | | |
| L1: +L_ord | | | |
| L2: +L_cont | | | |
| L3: +L_boundary | | | |

---

## 11. 绘图工具与规范

| 项目 | 选择 |
|------|------|
| 数据图 | Matplotlib 3.x + Seaborn |
| 示意图 | draw.io / TikZ (LaTeX) / Figma |
| 字体 | Arial/Helvetica, ≥8pt |
| 配色 | ColorBrewer Set2 (色盲友好) |
| 格式 | PDF (矢量) + PNG (300 dpi raster 备份) |
| 热力图 | jet → viridis (色盲友好, MedIA 推荐) |
| 统计标注 | `* p<0.05, ** p<0.01, *** p<0.001, n.s. 不显著` |

## 12. MedIA 图表规范 (检查清单)

- [ ] 所有文字 ≥8pt (打印后 ≥2mm)
- [ ] 颜色在灰度打印下可区分 (viridis colormap)
- [ ] 坐标轴均有标签和单位
- [ ] 图注详细到可脱离正文理解 (≥50 词)
- [ ] 无截图/低分辨率 raster 图
- [ ] 图表编号与正文引用严格对应
- [ ] 所有缩写在图注中首次出现时展开
- [ ] 数据图附带误差棒 (95% CI 或 ±1 std)

## 13. 生成顺序

| 阶段 | 图表 | 依赖 |
|------|------|------|
| 实验完成后 | Tab 1, Tab 2, Tab 3, Fig 3 | 全部指标 |
| 实验完成后 | Fig 4, Fig 5 | EDL u, s_sym |
| 实验完成后 | Fig 6, Fig 8 | 模型权重 |
| 随时 | Fig 1, Fig 2, Fig 7 | 架构文档 (无数据依赖) |
| 终稿前 | Fig S1-S3, Tab 4 | 完整结果 |

**建议**: Fig 1, Fig 2, Fig 7 (无数据依赖) 可在 Phase 5 实现时由 VizAgent 先行产出示意草稿, 实验完成后替换为最终版本。
