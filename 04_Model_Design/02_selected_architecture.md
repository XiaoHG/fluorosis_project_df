---
date: 2026-05-21
author: ModAgent
input_from: "04_Model_Design/01_architecture_variants/variant_1_full_symmamba.md, Human PI decision D2"
output_to: "04_Model_Design/02_selected_architecture.md"
status: confirmed_by_human
---

# 选定架构：Full SymMamba

## 选择理由

D2 决策（2026-05-21）：人类 PI 选定 Full SymMamba。Cross Scan Mamba 是论文标志性创新——首次将双侧对称性诊断原则编码入神经网络。V2（SFE）保留作为消融实验对比。

---

## 1. 架构总览

```
        Input: 512×256×3
              │
    ┌─────────┴─────────┐
    │  Patch Embedding   │  Conv2d(k=4,s=4) → 128×64×96
    └─────────┬─────────┘
              │
    ┌─────────┴──────────────────────────┐
    ▼                                     ▼
┌───────────────┐                  ┌───────────────┐
│  Arch Scan    │                  │  Cross Scan   │
│  Mamba ×4     │  Stage 1         │  Mamba ×4     │  Stage 1
└───────┬───────┘                  └───────┬───────┘
        ├──────── CGF1 ────────────────────┤
    ┌───┴───┐                        ┌───┴───┐
    │ Mamba │  Stage 2 (s=2)         │ Mamba │  Stage 2 (s=2)
    │ ×2    │                        │ ×2    │
    └───┬───┘                        └───┬───┘
        ├──────── CGF2 ────────────────────┤
    ┌───┴───┐                        ┌───┴───┐
    │ Mamba │  Stage 3 (s=2)         │ Mamba │  Stage 3 (s=2)
    │ ×2    │                        │ ×2    │
    └───┬───┘                        └───┬───┘
        ├──────── CGF3 ────────────────────┤
        └────────┬─────────────────────────┘
                 ▼
        Concat + GAP → z [768]
                 │
                 ▼
        EDL Head: Linear(768,256)→ReLU→Linear(256,4)→exp()+1
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Dirichlet α        s_sym (对称性得分)
   → u, b_k, pred
```

## 2. 输入与预处理

| 项目 | 规格 |
|------|------|
| 输入尺寸 | 512×256×3 |
| 归一化 | ImageNet mean/std |
| 增强 | 见 `05_Exp_Design/02_augmentation.md` 完整管线 |

## 3. 骨干网络

### 3.1 Patch Embedding
Conv2d(3,96,k=4,s=4) → BN → GELU → 128×64×96

### 3.2 Arch Scan
128×64×C → (128×64)×C 序列 → Mamba [d_state=16,d_conv=4,expand=2]

| Stage | Blocks | 输出 | 通道 |
|-------|--------|------|------|
| 1 | Mamba×4 | 128×64 | 96 |
| 2 | Mamba×2+Conv(s=2) | 64×32 | 192 |
| 3 | Mamba×2+Conv(s=2) | 32×16 | 384 |

### 3.3 Cross Scan
左右半区 split → 共享权重 Mamba → Concat。Symmetry Head: `s_sym = cos(GAP(left), GAP(flip(right)))`

### 3.4 CGF 融合
`Gate = σ(Linear(Concat(GAP(f_arch), GAP(f_cross)), C))` , `f_fused = Gate⊙f_arch + (1-Gate)⊙f_cross`

## 4. EDL Head

```
GAP(arch⊕cross) [768] → Linear → 256 → ReLU+Dropout(0.3) → Linear → 4 → exp()+1 → α
S=Σα_k, b_k=(α_k-1)/S, u=4/S, pred=argmax(α)
```

## 5. 损失函数

L = L_EDL + 0.1·L_ord + 0.05·L_cont + 0.01·L_boundary

- L_EDL: 标准 Dirichlet 证据损失
- L_ord: 有序正则化（禁止信念跳跃）
- L_cont: 有序对比损失（τ=0.07）
- L_boundary: 边界不确定性先验

## 6. 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW, weight_decay=1e-4 |
| LR | backbone 1e-4, EDL head 1e-3 |
| Schedule | CosineAnnealing + 5 epoch warmup |
| Batch | 16 |
| Epochs | 100 (early stop patience=20, monitor=val QWK) |
| 5-fold CV | 160 train / 40 val, 分层 |
| Dropout | 0.1 (Mamba), 0.3 (head), DropPath 0.1 |
| AMP | FP16 |

不确定性阈值: 扫描 θ∈[0.1,0.9], 选 SDR≥0.95 下 Retention Rate 最大。

## 7. 参数量

| 项目 | 值 |
|------|-----|
| 总参数 | ~17M |
| FP32 | ~68 MB |
| INT8 | ~17 MB |
| 推理 (RTX 3090) | ~8 ms |
| 推理 (CPU) | ~50 ms |

## 8. 可解释性

| 对象 | 来源 | 含义 |
|------|------|------|
| b_k | EDL Head α | 逐级证据 |
| u | EDL Head S | 不确定性 → 转诊 |
| s_sym | Cross Scan Symmetry Head | 对称性得分 |
| Arch Attention | Arch Scan 末层 | 沿牙弓注意力 |
| Cross Diff | CGF3 前左右特征差 | 不对称定位 |
| Grad-CAM | CGF3 输出 | 决策依据 |
| t-SNE | GAP 后 z | 特征有序性 |
