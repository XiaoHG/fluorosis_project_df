---
date: 2026-05-21
author: ExpDesignAgent
input_from: "03_Innovation/04_tech_roadmap.md, 04_Model_Design/02_selected_architecture.md, 04_Model_Design/01_architecture_variants/variant_2_symmamba_lite.md"
output_to: "05_Exp_Design/05_ablation_plan.md"
status: draft
---

# 消融实验矩阵

## 1. 实验设计原则

- 逐组件验证: 每次仅改变一个组件, 其他保持不变
- 所有消融使用同一 5-fold split, 报告 QWK ± std
- 每个实验可独立运行, 无依赖顺序
- 预期结果基于文献和经验, 用于判定结果是否合理

## 2. 消融实验矩阵

### E0: 纯 Baseline (无任何创新组件)

| 组件 | 状态 |
|------|------|
| Backbone | ResNet50 (ImageNet) |
| Head | Linear → 4 |
| Loss | CrossEntropy |
| SSL 预训练 | 无 |
| 对称性 | 无 |
| 有序约束 | 无 |
| 不确定性 | 无 |

**目的**: 确立性能下界。**预期 QWK**: ~75%

---

### E1: + EDL Head (验证 EDL 孤立效果)

| 组件 | 状态 |
|------|------|
| Backbone | ResNet50 (ImageNet) |
| Head | **EDL Head** (Linear→256→ReLU→Linear→4→exp()+1) |
| Loss | **L_EDL only** (无 L_ord, L_cont, L_boundary) |
| SSL | 无 |
| 对称性 | 无 |

**目的**: 验证标准 EDL (Sensoy 2018) 在氟斑牙上的效果, 排除有序约束的贡献。

**对比 E0**: 预期 QWK 持平或微增 (~76%), 但获得不确定性输出 (u)。关键看 u 是否在 Mild/Moderate 上自然升高。

**预期 QWK**: ~76%

---

### E2: + Ordinal Constraints (验证有序约束的增量)

| 组件 | 状态 |
|------|------|
| Backbone | ResNet50 (ImageNet) |
| Head | EDL Head |
| Loss | L_EDL + **L_ord (0.1) + L_cont (0.05) + L_boundary (0.01)** |
| SSL | 无 |

**目的**: 验证三个有序约束对 EDL 的增量贡献。

**对比 E1**: 预期 QWK 提升 3-5pp (有序约束抑制跨级误判)。同时验证 u 是否在相邻类边界达峰。

**预期 QWK**: ~80%

**子消融 (可选, 如时间允许)**:
- E2a: L_EDL + L_ord only
- E2b: L_EDL + L_ord + L_cont
- E2c: L_EDL + L_ord + L_cont + L_boundary (full E2)

---

### E3: + Arch Scan (Mamba 骨干替换)

| 组件 | 状态 |
|------|------|
| Backbone | **SymMamba — Arch Scan only** (10 Mamba blocks, 3 stages) |
| Head | EDL Head (含有序约束) |
| Loss | L_full (= E2) |
| Cross Scan | **无** |
| SSL | 无 |

**目的**: 验证 Mamba/SSM 序列建模 vs ResNet50 CNN。Arch Scan 沿牙弓方向扫描, 捕获釉质纹理的序列特征。

**对比 E2**: 预期 QWK 提升 2-3pp。Mamba 的序列建模天然适合牙弓沿线的逐牙检查范式。

**预期 QWK**: ~82%

---

### E3a: + SFE (Symmetry Feature Extractor, V2 对称性模块)

| 组件 | 状态 |
|------|------|
| Backbone | SymMamba — Arch Scan only (10 Mamba blocks) |
| Symmetry | **SFE** (参数-free: 左右半区特征均值/方差/余弦差异) |
| Fusion | Adaptive Sym Fusion (可学习权重 α) |
| Head | EDL Head (含有序约束) |
| Loss | L_full + L_sym (BCE, λ=0.05) |
| SSL | 无 |

**目的**: 验证 D2 保留的 V2 (SymMamba-Lite) SFE 模块。无额外可训参数，仅统计特征差异编码对称性。与 E3 (无对称性) 和 E4 (可学习 Cross Scan) 三级对比。

**预期 QWK**: ~83%

---

### E4: + Cross Scan (对称性编码, 关键消融)

| 组件 | 状态 |
|------|------|
| Backbone | SymMamba — **Arch Scan + Cross Scan** (完整 SymMamba) |
| Fusion | **CGF ×3** |
| Symmetry Head | **s_sym = cos(GAP(left), GAP(flip(right)))** |
| Head | EDL Head (含有序约束) |
| Loss | L_full |
| SSL | 无 |

**目的**: 核心消融 — 验证 Cross Scan 对称性编码的增量贡献。这是论文方法的核心差异化组件。

**对比 E3**: 这是最重要的对比。预期 QWK 提升 2-4pp, 且对称性得分 s_sym 与 Dean 标签在氟斑牙样本上呈正相关 (s_sym > 0.5)。

**预期 QWK**: ~85%

**额外分析**:
- 统计 E3 vs E4 在 Normal 类上的 Accuracy (正常牙齿双侧高度对称, Cross Scan 应更准)
- 检查 s_sym 分布: 氟斑牙样本是否确实高于非对称病变 (如有)
- 如 E4 vs E3 无显著提升 → Cross Scan 无效 → 降级为 V2 (SFE)

---

### E5: + Oral SSL 预训练

| 组件 | 状态 |
|------|------|
| Backbone | **Oral SSL 预训练** (COde 50K + Oral Diseases 13K + AlphaDent 1.3K → DINOv2) |
| 其他 | 同 E4 (完整 SymMamba + Ord EDL) |

**目的**: 验证口腔域 SSL 预训练 vs ImageNet 预训练的增益。

**对比 E4**: 预期 QWK 提升 1-2pp, 主要在牙面纹理和光照变化的鲁棒性上。如果提升 <1pp, SSL 阶段可降级为备选。

**预期 QWK**: ~87%

**子分析**:
- Linear probing Acc (SSL backbone frozen, 仅训 Linear 头) 作为预训练质量的中间检查
- 如果 linear probing Acc < 55%, SSL 预训练视为无效

---

### E6: 完整方案 (= Ours)

同 E5, 即完整论文方法。E6 本身不是独立消融实验, 而是前述消融链的终点。

**预期 QWK**: ≥87%, SDR ≥95%

## 3. 损失函数消融 (L-系列)

在 E4 (完整 SymMamba) 上对损失函数做子消融:

| 实验 | L_EDL | L_ord | L_cont | L_boundary | 目的 |
|------|-------|-------|--------|------------|------|
| L0 | ✓ | | | | 标准 EDL |
| L1 | ✓ | ✓ | | | +有序正则化 |
| L2 | ✓ | ✓ | ✓ | | +有序对比 |
| L3 | ✓ | ✓ | ✓ | ✓ | **完整** (= E4) |

**目的**: 区分三个有序约束各自的贡献。L_boundary 权重极低 (0.01), 预期对 QWK 影响微弱但影响 u 校准 (ECE)。

**预期**: L0→L1 提升最大 (~2pp QWK), L1→L2 提升 ~1pp, L2→L3 主要改善 ECE (u 在边界达峰)。

## 4. CGF 消融 (F-系列)

在 E4 上对融合策略做子消融:

| 实验 | 融合方式 | 目的 |
|------|----------|------|
| F0 | Concat(f_arch, f_cross) | 简单拼接 (baseline fusion) |
| F1 | f_arch + f_cross | 直接相加 |
| F2 | **CGF (Gate)** | 完整方案 (= E4) |

**目的**: 验证门控融合的必要性。

## 5. 消融实验总表

| ID | 实验 | Backbone | Head | Loss | Cross Scan | SSL | 预期 QWK | 验证重点 |
|----|------|----------|------|------|------------|-----|----------|----------|
| E0 | Baseline | ResNet50 | Linear | CE | ✗ | ✗ | ~75% | 下界 |
| E1 | +EDL | ResNet50 | EDL | L_EDL | ✗ | ✗ | ~76% | u 分布 |
| E2 | +Ord | ResNet50 | EDL | L_full | ✗ | ✗ | ~80% | 有序约束增益 |
| E3 | +Arch | Arch Mamba | EDL | L_full | ✗ | ✗ | ~82% | SSM vs CNN |
| E3a | +SFE(V2) | Arch Mamba+SFE | EDL | L_full+L_sym | ✗(SFE) | ✗ | ~83% | 参数-free 对称性 |
| E4 | **+Cross** | **Full SymMamba** | EDL | L_full | **✓** | ✗ | ~85% | **对称性增量 (核心)** |
| E5 | +SSL | Full SymMamba | EDL | L_full | ✓ | ✗ | ~86% | 域预训练增益 |
| E6 | **Full** | Full SymMamba | EDL | L_full | ✓ | **✓** | **~87%** | **完整方案** |

## 6. 消融报告模板

每个消融实验的报告包含:

```
### EX: [名称]
- QWK: [值] ± [std]
- SDR(θ*): [值] @ Retention = [值]
- ECE: [值]
- vs 前一级: ΔQWK = [值], 显著 (p<0.05)
- 结论: [一句话]
```

## 7. 执行优先级

| 优先级 | 实验 | 理由 |
|--------|------|------|
| P0 | E0, E4, E6 | Baseline + 核心消融 + 完整方案 (论文必备) |
| P1 | E3, E3a, E5 | Arch Scan 增量 + SFE vs Cross Scan + SSL 增量 |
| P2 | E1, E2 | EDL 逐级贡献 |
| P3 | L0-L3, F0-F2 | 子组件消融 (放入 Supplementary 或正文) |

如时间紧张, 至少完成 P0+P1 (7 个实验: E0, E3, E3a, E4, E5, E6), L/F 系列放入 Supplementary。
