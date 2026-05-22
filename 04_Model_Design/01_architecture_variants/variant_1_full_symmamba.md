---
date: 2026-05-21
author: ModAgent
input_from: "03_Innovation/03_selected_idea.md, 03_Innovation/04_tech_roadmap.md, 01_Knowledge_Base/data_description.md"
output_to: "04_Model_Design/01_architecture_variants/variant_1_full_symmamba.md"
status: draft
---

# Variant 1: Full SymMamba — 完整双路径对称性感知 Mamba 网络

## 1. 架构概述

Full SymMamba 将氟斑牙诊断中"沿牙弓逐牙检查"和"左右对称对比"两个临床行为完整编码为两条独立的 Mamba 扫描路径，通过三层交叉门控融合实现特征交互，最终由 Ordinal-aware EDL 头输出有序证据分布。

```
        Input: 512×256×3
              │
    ┌─────────┴─────────┐
    │  Patch Embedding   │  Conv2d(k=4,s=4) → 128×64×C1 (C1=96)
    └─────────┬─────────┘
              │
    ┌─────────┴──────────────────────────┐
    │                                     │
    ▼                                     ▼
┌───────────────┐                  ┌───────────────┐
│  Arch Scan    │                  │  Cross Scan   │
│  Mamba ×4     │                  │  Mamba ×4     │
│  (沿行序列)    │                  │  (左右交叉)   │
│  Stage 1      │                  │  Stage 1      │
└───────┬───────┘                  └───────┬───────┘
        │                                  │
        ├──────── CGF1 ────────────────────┤
        │                                  │
    ┌───┴───┐                        ┌───┴───┐
    │ Mamba │                        │ Mamba │
    │ ×2    │  Stage 2               │ ×2    │  Stage 2
    │ s=2   │                        │ s=2   │
    └───┬───┘                        └───┬───┘
        │                                  │
        ├──────── CGF2 ────────────────────┤
        │                                  │
    ┌───┴───┐                        ┌───┴───┐
    │ Mamba │                        │ Mamba │
    │ ×2    │  Stage 3               │ ×2    │  Stage 3
    │ s=2   │                        │ s=2   │
    └───┬───┘                        └───┬───┘
        │                                  │
        ├──────── CGF3 ────────────────────┤
        │                                  │
        └────────┬─────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │  Concat + GAP  │  → 2×C3 = 768
        └───────┬────────┘
                │
                ▼
        ┌────────────────┐
        │   EDL Head      │  Linear(768,256)→ReLU→Linear(256,4)→exp()+1
        └───────┬────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
   Dirichlet α       s_sym (对称性相似度)
   → u, b_k, pred
```

## 2. 输入层与预处理

| 项目 | 规格 |
|------|------|
| 输入尺寸 | 512×256×3 (RGB) |
| 归一化 | ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| 在线增强 | RandAugment(N=2, M=9), ColorJitter(0.2,0.2,0.2,0.1), 随机水平翻转(50%) |
| 在线预处理 | CutMix(α=0.5, 概率 50%) |

## 3. 骨干网络

### 3.1 Patch Embedding

```
Conv2d(in=3, out=96, kernel_size=4, stride=4)
→ BN → GELU
→ 输出: 128×64×96
```

stride=4 在 512×256 图像上产出 128×64 特征图，保留沿牙弓水平方向 128 列的序列分辨率以捕获跨牙面特征。

### 3.2 Arch Scan Branch

沿行方向将特征图展开为序列，模拟沿牙弓曲线逐牙扫描：

```
128×64×C → reshape → (128×64)×C 序列 [沿行方向]
         → Mamba Block ×4 [d_state=16, d_conv=4, expand=2]
         → reshape → 128×64×C
```

Mamba 参数: d_model=C, d_state=16, d_conv=4, expand_factor=2。3 个 stage，通道: C1=96 → C2=192 → C3=384。Stage 间通过 2×2 卷积（stride=2）下采样。

### 3.3 Cross Scan Branch

将特征图沿垂直中线分为左右半区，交叉扫描以对比双侧牙面：

```
128×64×C → split → left [0:64]×C, right [64:128]×C
                    │                    │
                    ▼                    ▼
              Mamba Block ×4      Mamba Block ×4
              (左→右方向)          (右→左方向)
                    │                    │
                    └────────┬───────────┘
                             ▼
                    Concat → 128×64×C
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              128×64×C         Symmetry Head:
              (特征图)          GAP(left)·GAP(flip(right))
                                ─────────────────────────
                                ∥left∥·∥right∥
                                → s_sym ∈ [-1,1]
```

左右序列共享 Mamba 权重（体现解剖对称性）。Symmetry Head 输出余弦相似度作为对称性得分。

### 3.4 Stage 配置

| Stage | Arch Scan | Cross Scan | 输出尺寸 | 通道数 |
|-------|-----------|------------|----------|--------|
| 1 | Mamba ×4 | Mamba ×4 | 128×64 | 96 |
| 2 | Mamba ×2 + Conv(s=2) | Mamba ×2 + Conv(s=2) | 64×32 | 192 |
| 3 | Mamba ×2 + Conv(s=2) | Mamba ×2 + Conv(s=2) | 32×16 | 384 |

## 4. 融合模块

### CGF (Cross Gated Fusion) ×3

```
f_arch: B×H×W×C
f_cross: B×H×W×C

Gate = σ(Linear(Concat(GAP(f_arch), GAP(f_cross)), C))
f_fused = Gate ⊙ f_arch + (1-Gate) ⊙ f_cross
```

Gate ∈ [0,1]^C：每个通道独立决定 Arch/Cross 分支的信任权重。3 个 CGF 的 Gate 权值独立，训练后可分析不同层级的两分支贡献。

## 5. 任务头

### EDL Classification Head

```
GAP(f_arch ⊕ f_cross) → z [768]
                         │
                    Linear(768, 256)
                         │ ReLU + Dropout(0.3)
                    Linear(256, 4)
                         │ exp() + 1
                         ▼
                    α = [α0, α1, α2, α3]
```

| 符号 | 公式 | 含义 |
|------|------|------|
| S | Σα_k | Dirichlet 强度 |
| b_k | (α_k - 1) / S | 第 k 类信念质量 |
| u | 4 / S | 全局不确定性 |
| pred | argmax(α) | 预测类别 |
| s_sym | cos(left, right) | 对称性得分 |

## 6. 损失函数

总损失 = L_EDL + 0.1·L_ord + 0.05·L_cont + 0.01·L_boundary

**L_EDL**: 标准 Dirichlet 证据损失，最小化信念质量与 one-hot 标签的 MSE 差 + 压制非目标类证据。

**L_ord** (有序正则): 惩罚信念质量 b_k 在类别间跳变，强制邻级平滑过渡。

**L_cont** (有序对比, τ=0.07): 拉近 |y_i-y_j|≤1 样本的特征，推远跨级样本。

**L_boundary** (边界不确定性): 对邻级样本，允许 u 适度升高。

## 7. 训练配置

| 参数 | 值 | 理由 |
|------|-----|------|
| 优化器 | AdamW | Mamba + EDL 联合训练 |
| LR | backbone 1e-4, EDL head 1e-3 | Head 从头训练需更大 LR |
| LR schedule | CosineAnnealing + 5 epoch warmup | 稳定早期训练 |
| Weight decay | 1e-4 | 200 张数据需强正则 |
| Batch size | 16 | RTX 3090/4090 24GB |
| Epochs | 100 (early stop patience=20) | 充分收敛 |
| 5-fold CV | 每折 160 train / 40 val | 分层保持类别比例 |
| Dropout | 0.1 (Mamba), 0.3 (EDL head) | 防过拟合 |
| DropPath | 0.1 | Stochastic depth |
| Mixed precision | AMP (FP16) | Mamba 支持 FP16 |
| 平衡策略 | 无 | 4 类各 50 张已平衡 |

## 8. 预期参数量与推理速度

| 项目 | 估算 |
|------|------|
| Arch Scan (Mamba ×10) | ~8M |
| Cross Scan (Mamba ×10) | ~8M |
| Patch Embedding | ~0.1M |
| CGF ×3 | ~0.3M |
| EDL Head | ~0.4M |
| **总计** | **~17M** |
| FP32 模型大小 | ~68 MB |
| INT8 量化后 | ~17 MB |
| 推理速度 (RTX 3090) | ~8 ms/张 |
| 推理速度 (CPU, ONNX) | ~50 ms/张 |

## 9. 可解释性出口

| 可视化对象 | 来源层 | Hook 位置 | 医学解释 |
|-----------|--------|-----------|---------|
| 逐级证据 b_k | EDL Head α 输出 | forward 最后一步 | 每类诊断信心 |
| 不确定性 u | EDL Head S 计算 | forward 最后一步 | 标记需转诊的困难病例 |
| 对称性得分 s_sym | Cross Scan Symmetry Head | Cross Scan 输出 | 双侧病变对称性评估 |
| Arch Attention Map | Arch Scan 末层 Mamba SSM hidden state | Mamba.forward | 哪些牙位被重点分析 |
| Cross Diff Map | 左右特征差值图 | CGF3 前 | 双侧不对称区域定位 |
| Grad-CAM | CGF3 输出层 | 融合后特征 | 最终决策依赖的空间区域 |
| t-SNE | GAP 后 z [768] | EDL Head 输入 | 4 级特征有序排列验证 |

## 10. 优缺点

**优势**：完整实现创新点全部组件，Cross Scan 独立路径充分学习对称性，CGF 自动调整两分支贡献，可解释性出口丰富。

**劣势**：参数量 ~17M，Cross Scan Mamba 在小数据上可能训练不稳定，左右硬分区假设图像中牙齿严格居中。
