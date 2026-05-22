---
date: 2026-05-21
author: ModAgent
input_from: "03_Innovation/04_tech_roadmap.md, 04_Model_Design/01_architecture_variants/"
output_to: "04_Model_Design/04_training_config.md"
status: draft
---

# 训练配置

## 统一配置（两变体通用部分）

### 数据

| 参数 | 值 |
|------|-----|
| 总样本 | 200 (4 类各 50) |
| 图像尺寸 | 512×256 (W×H) |
| 5-fold CV | 每折 160 train / 40 val, 分层保持类别比例 |
| 随机种子 | 42 |

### 数据增强

详见 `05_Exp_Design/02_augmentation.md`。核心管线: 几何增强 (水平翻转+旋转+裁剪) → 颜色增强 (ColorJitter) → 纹理增强 (GaussianBlur+Sharpness) → Mixing (CutMix)。

| 关键参数 | 值 |
|------|------|
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05 |
| CutMix | α=0.5, p=50% |
| 归一化 | ImageNet mean/std |

### 优化器与调度

| 参数 | V1 (Full SymMamba) | V2 (SymMamba-Lite) |
|------|-----|-----|
| 优化器 | AdamW (β=0.9, 0.999) | 同 |
| LR backbone | 1e-4 | 1e-4 (统一) |
| LR EDL head | 1e-3 | 1e-4 (统一) |
| Weight decay | 1e-4 | 1e-4 |
| Warmup | 5 epochs linear | 同 |
| Schedule | CosineAnnealing → min 1e-6 | 同 |

### 训练控制

| 参数 | V1 | V2 |
|------|-----|-----|
| Epochs | 100 | 100 |
| Early stop | val QWK, patience=20 | val QWK, patience=15 |
| Batch size | 16 | 16 |
| Gradient clip | — | max_norm=1.0 |
| Mixed precision | AMP (FP16) | 同 |

### 正则化

| 参数 | V1 | V2 |
|------|-----|-----|
| Dropout (Mamba) | 0.1 | 0.2 |
| Dropout (EDL head) | 0.3 | 0.3 |
| DropPath | 0.1 | 0.1 |

### 不确定性阈值校准

每折验证集上扫描 θ ∈ [0.1, 0.9] (step=0.05)，选 SDR ≥ 0.95 下 Retention Rate 最大的 θ。若无法满足，选 SDR 最大的 θ。

### 硬件估算

| 项目 | V1 | V2 |
|------|-----|-----|
| 推荐 GPU | RTX 3090/4090 (24GB) | 同 |
| 每折时间 | ~2h | ~1.5h |
| 5-fold 总时间 | ~10h | ~8h |
| SSL 预训练 | ~24h (A100, 一次性) | 同 |

### 监控

- Train Loss (各组件): 每 50 step
- Val QWK, SDR, Macro F1, ECE: 每 epoch
- Confusion Matrix: 每 epoch
- 最优模型保存: val QWK 最高

### 复现性清单

- [ ] 固定随机种子 (Python, NumPy, PyTorch, CUDA)
- [ ] 记录 5 折划分索引
- [ ] 记录预训练权重路径
- [ ] 记录 WandB run ID
