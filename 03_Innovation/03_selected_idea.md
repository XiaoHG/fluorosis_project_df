---
date: 2026-05-21
author: InnoAgent (refined with Human PI feedback)
input_from: "02_Literature/ 全部产出, 03_Innovation/01_radar_chart.txt, 03_Innovation/02_idea_proposals/, Human PI discussion (2026-05-21)"
output_to: "03_Innovation/03_selected_idea.md"
status: final
---

# 选定创新点：对称性感知 Mamba 与有序证据学习的氟斑牙自动化分级诊断

## 1. 创新点标题

**Symmetry-Aware Mamba with Ordinal-aware Evidential Deep Learning for Automated Dental Fluorosis Grading and Screening-Ready Diagnosis**

## 2. 一句话概括

将牙科医生"双侧对比、由轻到重、不确定时转诊"的诊断逻辑编码入神经网络——SymMamba 提供对称性结构先验，Ordinal-aware EDL 输出有序证据分布和不确定性估计，SDR 评估筛查安全性，最终以结构化诊断报告呈现临床可用的分级结果。

## 3. 对应缺口

| 缺口编号 | 缺口描述 | 本方案如何填补 |
|----------|----------|---------------|
| G1 | 有序回归完全空白 | Ordinal-aware EDL：Dirichlet 证据分布施加有序约束，类间证据平滑过渡 |
| G2 | 自监督预训练空白 | Oral-domain SSL：COde 50K + Oral Diseases 13K + AlphaDent 1.3K → DINOv2 预训练 |
| G3 | 注意力机制无临床先验引导 | SymMamba Cross Scan：编码 Cutress & Suckling (1990) 双侧对称性诊断原则 |
| G6 | 可解释性无定量验证 | EDL 逐级证据量 + 不确定性可视化 + SDR 提供筛查安全性量化 |
| G7 | 多尺度特征融合专门化 | Arch Scan 沿牙弓曲线的多尺度序列建模 + Cross Scan 左右对比 |
| G4 | 标注噪声未建模 | EDL 天然输出预测不确定性，标注噪声样本表现为高 u |
| G5 | 评估指标不统一 | 建立 SDR + QWK + 校准误差的三维评估协议 |

## 4. 核心创新三支柱

### 支柱 1：SymMamba — 对称性感知双路径 Mamba 编码器

**背景**：现有所有氟斑牙方法（MLTrMR、FusionDentNet、LD2Net、Mwinc-Mamba）将牙面图像作为普通图像处理，完全忽略氟斑牙的核心诊断原则——**双侧对称性**。Cutress & Suckling (1990) 明确指出："氟斑牙对称性分布于同名牙，双侧浑浊分布模式的一致性是区别于其他釉质病变的关键鉴别特征。"

**设计**：两个互补的 Mamba 扫描路径，在特征层级联融合。

| 扫描路径 | 方向 | 功能 | 对应临床行为 |
|----------|------|------|-------------|
| **Arch Scan** | 沿牙弓曲线（水平方向）逐牙扫描 | 捕获单颗牙齿的釉质纹理特征（白垩斑、着色、缺损），序列建模 Dean 分级在不同牙位上的表现 | 医生逐牙检查唇面病变 |
| **Cross Scan** | 左右镜像交叉扫描（图像左半区 ↔ 右半区） | 计算双侧同名牙区域的特征差异，将"对称性"编码为可学习的特征表示 | 医生对比双侧牙面对称性 |

**为什么是 Mamba 而非 Transformer**：
- 线性复杂度 O(N)：512×256 图像展开后 token 数多，ViT 的 O(N²) 在小样本下更不稳定
- SSM 的序列建模天然适合 Arch Scan 的"沿牙弓逐牙"范式
- Mwinc-Mamba (Xu 2025) 已在骨骼氟中毒 X 光上验证 SSM 有效性，本方案将其适配至氟斑牙自然光照片并增加 Cross Scan

**融合策略**：Arch Scan 和 Cross Scan 在 3 个特征层级输出特征，通过交叉门控融合（CGF）模块交互，最终合并进入 EDL 分类头。

### 支柱 2：Ordinal-aware EDL — 有序证据深度学习

**背景**：标准 EDL 输出 Dirichlet 分布的浓度参数 α = (α₀, α₁, α₂, α₃)，其中 α_k = e_k + 1，e_k 为第 k 类的证据量。但标准 EDL 不约束证据在类别间的分布——可能出现"Normal 证据 0.9、Severe 证据 0.8、Mild 证据 0.1"这种违反有序性的情况。

**设计**：在三层约束下训练 EDL：

**约束 1 — 有序正则化（Ordinal Regularization）**
```
L_ord = Σ_{k=1}^{K-2} max(0, b_{k-1} + b_{k+1} - 2·b_k)
```
强制信念质量 b_k 在相邻类别间平滑过渡，禁止"跳跃式"证据分布。

**约束 2 — 边界不确定性最大化（Boundary Uncertainty Prior）**
对于真实标签为 k 的样本，期望不确定性 u 在 k 与 k±1 的边界处达到峰值。通过在训练中添加一个微弱先验：当预测的 α 分布在相邻类上分布均匀时（即模型在类边界上犹豫），不施加过度惩罚。

**约束 3 — 有序对比损失（Ordinal Contrastive Loss）**
拉近相邻等级样本的特征，推远跨等级样本：
```
L_cont = -log[ Σ_{j: |y_j - y_i| ≤ 1} exp(sim(z_i, z_j)/τ) / Σ_{all j} exp(sim(z_i, z_j)/τ) ]
```

**与标准 EDL + CORAL 的区别**：
- CORAL 做的是对 logits 施加有序约束（K-1 个二分类器）
- 本方案做的是对 **Dirichlet 证据分布本身**施加有序约束——证据量在类别间平滑分布，不确定性在边界处达峰

**不确定性驱动的转诊机制**：
当 u > 阈值 θ（在验证集上校准后确定）时，模型拒绝自动判级，标记为"建议转诊临床医生复核"。

### 支柱 3：SDR — 重度检出率

采用 **SDR（Severe Detection Rate）** 作为核心创新评估指标：

```
SDR = Recall(Severe | model is confident)
    = TP_severe / (TP_severe + FN_severe)
```

其中 "confident" 定义为 u ≤ θ（即模型不请求转诊的样本子集）。

**SDR 的临床含义**：在筛查系统自主判级的病例中，重度氟斑牙被成功检出的比例。SDR = 100% 意味着没有一个重度患者被漏掉——这是筛查安全性的直接度量。

**与现有指标的差异化**：

| 维度 | Accuracy / F1 | SDR |
|------|-------------|-----|
| 评估对象 | 模型分类能力 | 筛查系统安全性 |
| 临床含义 | "模型判对了吗" | "重症全抓住了吗" |
| 现有文献 | 全部使用 | 零 |
| 与 EDL 的关系 | 无关 | EDL 的 u 决定何时转诊 → 直接影响 SDR |

**补充指标**：
- **QWK（Quadratic Weighted Kappa）**：作为与 MLTrMR 等 baseline 可比较的主指标
- **UR-Acc（Uncertainty-Referred Accuracy）**：放入 Supplementary Material 作为部署效率参考
- **ECE（Expected Calibration Error）**：评估 EDL 不确定性估计的校准质量

## 5. 创新点 4（辅助）：EDL 驱动的结构化诊断报告

### 5.1 设计思路

EDL 的 forward pass 天然产出诊断报告所需的所有数值——分级结果、逐级证据、全局不确定性。报告不依赖大语言模型生成，而是使用**规则模板 + 数据填充**的方式，零幻觉风险。

### 5.2 报告结构

```
==========================================
  氟斑牙自动化分级诊断报告
==========================================
Date: YYYY-MM-DD
Image ID: DF_XXXX_XXXX

--- 分级结果 ---
Dean 分级: [Mild/Moderate/...] ([0-3] 级)
信念质量: [b_k 值，如 0.81]

--- 不确定性评估 ---
全局不确定性: u = [0.00-1.00]
→ [u ≤ θ: "置信度高，结果可直接使用"]
→ [u > θ: "不确定性较高，建议转诊临床医生复核"]

--- 逐级证据 ---
Normal:    ▏▏  b₀ = [值]
Mild:      ▏▏  b₁ = [值]
Moderate:  ▏▏  b₂ = [值]
Severe:    ▏▏  b₃ = [值]

--- 对称性分析 ---
双侧牙面对比相似度: [Cross Scan 特征余弦相似度]
→ [≥0.9: "双侧牙面病变分布高度对称，符合氟斑牙特征"]
→ [<0.9: "双侧不对称，建议排除其他釉质病变"]

--- 临床建议 ---
[由规则引擎根据 Dean 分级 + u + 对称性生成]
示例: "Dean 2级，证据充分，双侧对称，支持氟斑牙诊断。
      建议结合Dean指数确诊，关注是否进展至点状缺损。"
```

### 5.3 技术实现

- **输入**：SymMamba 输出的特征向量 → EDL 头输出的 α 值 + Cross Scan 分支的对称性相似度
- **模板引擎**：Python 字符串模板，约 10 条规则覆盖所有分级-不确定性-对称性组合
- **不涉及 LLM**：无 API 调用，无自由文本生成，保证医学安全性

### 5.4 在论文中的定位

作为"Clinical Deployment" 或 "Diagnostic Reporting" 子节放在 Experiments 末尾或 Discussion 开头，展示模型从分类器到辅助诊断工具的转化路径。明确说明报告由规则模板生成，非 AI 自由撰写——与 MedIA 的 AI 使用合规性无冲突。

## 6. 医学合理性论证

### 6.1 为什么对称性编码是必须的

Cutress & Suckling (1990) 综述明确指出："氟斑牙对称性分布于同名牙，这是区别于创伤性/感染性釉质缺陷的关键特征。" 现有所有 DL 方法忽略这一临床知识——它们将 512×256 图像作为整体输入 CNN/ViT，没有架构级设计来对比左右牙面。SymMamba 的 Cross Scan 是**首次将对称性诊断原则编码入神经网络架构**。

### 6.2 为什么有序约束是必须的

Dean 0→1→2→3 是递进的病理过程。Mild 和 Moderate 的区分是临床最大难点（MLTrMR Fig.1f 证实非牙医无法可靠区分）。Ordinal-aware EDL 通过在证据空间施加有序正则化，使模型在 Mild/Moderate 边界上"知道自己不确定"——这不只是提高准确率，更是提供临床合理的错误模式（宁可给出"不确定"也不把 Mild 判成 Severe）。

### 6.3 为什么 SDR 是必须的

氟斑牙筛查的伦理底线是：**不能漏掉需要治疗的重度患者**。Dean 3 的患者需要干预，Dean 0-1 可以观察随访。一个把 Severe 判成 Mild 的模型在学术指标上扣 1 分，在临床场景里意味着一例本可避免的延误治疗。SDR 把这个伦理约束编码为评估指标。

## 7. 技术路线概述

```
阶段 A: 口腔域自监督预训练
  COde (50K) + Oral Diseases (13K) + AlphaDent (1.3K)
  → SSL (DINOv2 或 MAE) 预训练 ViT/CNN backbone
  → 学习牙面纹理、光泽、口腔光照变化的视觉先验

阶段 B: 氟斑牙域适应（可选）
  本项目 200 张氟斑牙图像
  → 第二阶 MIM 预训练（Masked Image Modeling on dental surfaces）
  → 学习氟斑牙特有的白垩色/褐色纹理

阶段 C: SymMamba + Ordinal-aware EDL 联合训练
  200 张 4 级标注数据
  → Arch Scan + Cross Scan 双路径特征提取
  → EDL 头输出 α 分布
  → L_total = L_EDL + λ1·L_ord + λ2·L_cont + λ3·L_boundary
  → 5-fold CV，验证集校准不确定性阈值 θ

阶段 D: 评估与报告
  → 主指标: SDR + QWK
  → 补充: UR-Acc, ECE, 混淆矩阵
  → 结构化诊断报告模板填充
```

## 8. 与现有 SOTA 的系统性比较

| 维度 | MLTrMR | FusionDentNet | LD2Net | Mwinc-Mamba | **本方案** |
|------|--------|---------------|--------|-------------|-----------|
| Backbone | ViT | CNN+Transformer | CNN (DSConv) | CNN+SSM | **SymMamba (双路径 SSM)** |
| 对称性建模 | 无 | 无 | 无 | 无 | **Cross Scan 左右对称对比** |
| 有序性建模 | 无 (交叉熵) | 无 (交叉熵) | 无 (交叉熵) | 无 (交叉熵) | **Ordinal-aware EDL** |
| 不确定性 | 无 | 无 | 无 | 无 | **Dirichlet u, 转诊机制** |
| 预训练 | 随机初始化 | ImageNet | ImageNet | 随机初始化 | **口腔域 SSL (64K images)** |
| 评估指标 | Acc, F1, QWK | Acc, F1 | Acc, F1 | Acc, F1 | **SDR + QWK + ECE** |
| 诊断报告 | 无 | 无 | 无 | 无 | **结构化报告输出** |
| 数据 | DFID 131 | DFID 131 | DFID 200 | SFXRay 80 | **200 张 (无额外标注)** |

## 9. 可行性评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 创新程度 | **5/5** | 对称性编码+有序EDL+SDR+诊断报告，四个组件在氟斑牙领域均为首次 |
| 实现难度 | **3/5** | Mamba 工具链成熟（mamba.py / Vim）；EDL 有现成 PyTorch 实现；报告为规则模板；主要工作在集成和调参 |
| 数据需求 | **已满足** | 200 张标注数据用于 fine-tune，64K 公开口腔数据用于预训练，无需额外标注 |
| 预期提升 | **显著** | QWK 82% → ≥85%；首次建立 SDR 基准；首次实现诊断报告输出 |
| 医学说服力 | **高** | 每个组件直接对应临床诊断逻辑，不是纯技术堆砌 |

## 10. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| Mamba 在 200 张小样本上训练不稳定 | 中 | Plan B: Cross Scan 降级为通道维度的左右特征差异计算（轻量对称性模块），保留对称性编码但减少可训参数 |
| EDL 的 uncertainty 未校准（u 始终很低或很高） | 中 | 在验证集上做 post-hoc temperature scaling 校准 ECE；参考 EDL 校准文献（Deng 2024） |
| 口腔域 SSL 预训练后 fine-tune 无提升 | 低 | 降级为 ImageNet 预训练 + 仅用氟斑牙数据做 MIM 第二阶段 |
| Cross Scan 与 Arch Scan 特征融合困难 | 中 | CGF 模块从简设计（特征拼接 + 1×1 conv），复杂化在后继消融实验中逐步增加 |

## 11. 对标文献

| 文献 | 局限 | 本方案差异化 |
|------|------|------------|
| MLTrMR (Wu 2024) | 随机初始化 ViT，交叉熵，无对称性 | SSL 预训练 + Ordinal EDL + Cross Scan |
| FusionDentNet (Gu 2024) | 两阶段，无不确定性，无可解释性定量 | 端到端 + EDL u + SDR 安全性评估 |
| LD2Net (Li 2026) | 纯 CNN，ImageNet 域差异大 | 口腔域 SSL + SSM 对称性编码 |
| Mwinc-Mamba (Xu 2025) | 骨骼病 X 光，单扫描路径 | 氟斑牙自然光适配 + 双路径扫描 |
| EDL 标准实现 (Sensoy 2018) | 无有序约束 | 在 Dirichlet 空间施加序数正则化 |

## 12. D1 决策记录

- **决策者**：人类 PI
- **日期**：2026-05-21
- **选定方向**：SymMamba + Ordinal-aware EDL + SDR + 结构化诊断报告
- **排除方向**：纯 MAE 预训练（已尝试失败）、解剖先验注意力（并入 Cross Scan 的对称性先验）、纯轻量化路线（留作后续工作）
