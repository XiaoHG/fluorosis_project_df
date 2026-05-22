---
date: 2026-05-21
author: ModAgent
input_from: "03_Innovation/03_selected_idea.md, 03_Innovation/04_tech_roadmap.md, 01_Knowledge_Base/data_description.md"
output_to: "04_Model_Design/01_architecture_variants/variant_2_symmamba_lite.md"
status: draft
---

# Variant 2: SymMamba-Lite — 单路径 Mamba + 对称性特征提取器

## 1. 架构概述

SymMamba-Lite 保留 Arch Scan 作为唯一 Mamba 编码路径，将 Cross Scan 精简为轻量 Symmetry Feature Extractor（SFE）——一个不参与序列建模、仅在特征层级计算左右统计差异的模块。SFE 在 3 个 stage 输出对称性差异特征，与 Arch 特征做 concat 融合后进入 EDL 头。

设计目标：在 200 张小数据集上更稳定训练，参数量控制在 Variant 1 的一半以内，同时保留对称性对比的诊断价值。

```
        Input: 512×256×3
              │
    ┌─────────┴─────────┐
    │  Patch Embedding   │  Conv2d(k=4,s=4) → 128×64×96
    └─────────┬─────────┘
              │
              ▼
    ┌─────────────────────────────────────────┐
    │           Arch Scan Branch               │
    │                                         │
    │  Stage 1: Mamba ×4 → 128×64×96    ──→ SFE1 (左右差分)
    │  Stage 2: Mamba ×2 + Down → 64×32×192 ──→ SFE2
    │  Stage 3: Mamba ×2 + Down → 32×16×384 ──→ SFE3
    └─────────────────┬───────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
      SFE1_diff    SFE2_diff    SFE3_diff
      [B,2×96]     [B,2×192]    [B,2×384]
         │            │            │
         └────────────┼────────────┘
                      ▼
         ┌───────────────────────┐
         │  Adaptive Sym Fusion   │  可学习权重 α1,α2,α3
         └───────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
   Arch GAP [384]          Sym Feat [768]
         │                       │
         └───────────┬───────────┘
                     ▼
              Concat → z [1152]
                     │
                     ▼
            ┌────────────────┐
            │   EDL Head      │  Linear(1152,256)→ReLU→Linear(256,4)
            └───────┬────────┘
                    │
            ┌───────┴────────┐
            ▼                ▼
       Dirichlet α       s_sym (标量)
```

## 2. 输入层与预处理

| 项目 | 规格 | 与 V1 差异 |
|------|------|-----------|
| 输入尺寸 | 512×256×3 (RGB) | 同 V1 |
| 归一化 | ImageNet mean/std | 同 V1 |
| 增强 | RandAugment(N=2,M=9), ColorJitter(0.2,0.2,0.2,0.1), 水平翻转(50%) | 同 V1 |
| CutMix | α=0.5, 50% 概率 | 同 V1 |

## 3. 骨干网络

### 3.1 Patch Embedding

Conv2d(k=4,s=4) → BN → GELU → 128×64×96。同 V1。

### 3.2 Arch Scan Branch (唯一序列建模路径)

| Stage | Mamba Blocks | 输出尺寸 | 通道数 |
|-------|-------------|----------|--------|
| 1 | ×4 | 128×64 | 96 |
| 2 | ×2 + Conv(s=2) | 64×32 | 192 |
| 3 | ×2 + Conv(s=2) | 32×16 | 384 |

Mamba 参数: d_state=16, d_conv=4, expand=2。共 10 个 Mamba block。

### 3.3 Symmetry Feature Extractor (SFE)

无参数对称性计算模块，在每个 stage 输出上执行：

```
输入: f_arch_i [B, H, W, C]

Step 1: Split — f_left = f[:, :, :W//2, :]
                  f_right = f[:, :, W//2:, :]

Step 2: Flip — f_right_flipped = flip(f_right, dim=2)

Step 3: Diff — diff = |f_left - f_right_flipped|

Step 4: Stats — mean_diff = mean(diff, dim=[1,2])  → [B, C]
                std_diff  = std(diff, dim=[1,2])    → [B, C]

Step 5: Output — sfe_i = Concat(mean_diff, std_diff) → [B, 2C]
```

SFE 无可训参数，纯粹计算左右半区的统计差异。这种"特征级对称性"编码了 Cross Scan 的临床意图，但避免了在小数据上训练第二条 Mamba 路径的风险。

**Symmetry Score**: s_sym = σ(Linear(Concat(sfe1, sfe2, sfe3), 1)) → [0,1]

### 3.4 Adaptive Symmetry Fusion

```
α_i = softmax(w_i)   [3 个可训标量]
s_feat = Σ α_i · Linear(sfe_i, 256)
       → [B, 768]
```

## 4. 融合模块

不使用 V1 的 CGF 门控融合，采用更简单的 Concat 融合：

```
z_arch = GAP(f_arch_3) → [B, 384]
z_sym  = s_feat        → [B, 768]
z      = Concat(z_arch, z_sym) → [B, 1152]
```

选择 Concat 的原因：Arch 特征（序列建模语义）和 Sym 特征（统计差异）语义差异大，Concat 保留各自信息空间，让后续 FC 层学习交互。

## 5. 任务头

```
z [1152] → Linear(1152, 256) → ReLU → Dropout(0.3)
         → Linear(256, 4) → exp() + 1 → α [4]

S = Σα_k,  b_k = (α_k-1)/S,  u = 4/S,  pred = argmax(α)
```

## 6. 损失函数

同 V1：L_total = L_EDL + 0.1·L_ord + 0.05·L_cont + 0.01·L_boundary

额外添加对称性辅助损失：

```
L_sym = BCE(s_sym, y_sym)  [λ=0.05]
```

y_sym 由 SFE 的左右差异统计值在训练 5 epoch 后通过 K-means(k=2) 自动赋值。

## 7. 训练配置

| 参数 | 值 | 与 V1 差异 |
|------|-----|-----------|
| 优化器 | AdamW | 同 |
| LR | 1e-4 (统一) | V1 分 backbone/head LR |
| LR schedule | CosineAnnealing + 5 epoch warmup | 同 |
| Weight decay | 1e-4 | 同 |
| Batch size | 16 | 同 |
| Epochs | 100 (early stop patience=15) | patience 更短 |
| Dropout | 0.2 (Mamba), 0.3 (EDL head) | Mamba dropout 高于 V1 |
| Mixed precision | AMP (FP16) | 同 |
| 梯度裁剪 | max_norm=1.0 | V1 无 |

## 8. 预期参数量与推理速度

| 项目 | 估算 | 与 V1 对比 |
|------|------|-----------|
| Arch Scan (Mamba ×10) | ~8M | V1 的一半 |
| SFE | ~0M | 无可训参数 |
| Adaptive Sym Fusion | ~0.4M | V1 无 |
| Patch Embedding | ~0.1M | 同 |
| EDL Head | ~0.5M | z 维度更大 |
| **总计** | **~9M** | V1: ~17M |
| FP32 大小 | ~36 MB | V1: 68 MB |
| INT8 大小 | ~9 MB | V1: 17 MB |
| 速度 (RTX 3090) | ~5 ms | V1: 8 ms |
| 速度 (CPU ONNX) | ~30 ms | V1: 50 ms |

## 9. 可解释性出口

| 可视化对象 | 来源层 | Hook 位置 | 医学解释 |
|-----------|--------|-----------|---------|
| 逐级证据 b_k | EDL Head α | forward 最后一步 | 同 V1 |
| 不确定性 u | EDL Head S | forward 最后一步 | 同 V1 |
| 对称性得分 s_sym | Adaptive Sym Fusion | step 3.3 | 定量对称性评估 |
| 左右差异图 ×3 | SFE diff (各 stage) | SFE step 3 | 定位双侧不对称区域 |
| Arch Attention Map | Arch Scan 末层 | Mamba.forward | 沿牙弓注意力分布 |
| Grad-CAM | Stage 3 输出 | 末层后 | 决策依据 |
| SFE 贡献权重 α_i | Adaptive Sym Fusion | softmax 后 | 哪一层对称性最重要 |
| t-SNE | GAP 后 z [1152] | EDL Head 输入 | 特征有序性 |

## 10. 优缺点

**优势**：
- 参数量 ~9M，约为 V1 的一半，训练更稳定
- SFE 无参数，对称性计算不会因数据量小而过拟合
- 单 Mamba 路径训练收敛更快，调试更容易
- 推理速度更快（5ms vs 8ms），更适合筛查场景
- 多级 SFE diff 图可直接可视化左右不对称位置

**劣势**：
- SFE 仅做统计级对称性（均值/方差差分），不如 Cross Scan Mamba 能学到非线性对称性模式
- 对称性信息仅通过 concat 注入最终表示，缺乏中间层级交互
- 如果氟斑牙对称性模式非常复杂（如非对称性病例），SFE 可能捕获不足
- 论文"创新度"略低于 V1（Cross Scan Mamba 才是完整创新承诺）
