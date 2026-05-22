---
date: 2026-05-21
author: ExpDesignAgent
input_from: "01_Knowledge_Base/data_description.md, 04_Model_Design/02_selected_architecture.md"
output_to: "05_Exp_Design/02_augmentation.md"
status: draft
---

# 数据增强策略

## 1. 增强原则

- 200 张样本量小, 增强是防过拟合关键
- 全部**在线增强** (训练时随机施加): 节省存储, 每 epoch 样本多样性更高
- 验证集仅归一化
- 增强保持氟斑牙病变的医学完整性: 颜色变化不改变诊断信号 (白垩色 vs 棕褐色)

## 2. 增强管线 (按施加顺序)

### 几何增强

| 增强 | 参数 | 概率 | 理由 |
|------|------|------|------|
| 水平翻转 | — | 50% | 牙齿自然对称, 翻转不改变 Dean 标签 |
| 随机旋转 | ±10° | 50% | 模拟拍摄角度偏差 |
| RandomResizedCrop | scale=(0.8,1.0), 512×256 | 50% | 模拟不同拍摄距离, 保持 2:1 宽高比 |

### 颜色增强

| 增强 | 参数 | 概率 | 理由 |
|------|------|------|------|
| Brightness | ±0.2 | 100% | 模拟不同光照 (自然光 vs 诊室灯) |
| Contrast | ±0.2 | 100% | 模拟不同相机设置 |
| Saturation | ±0.1 | 100% | 低幅度, 避免改变釉质色泽 |
| Hue | ±0.05 | 100% | 极低, 仅模拟轻微色温变化 |

### 纹理增强

| 增强 | 参数 | 概率 | 理由 |
|------|------|------|------|
| GaussianBlur | k=3, σ=(0.1,1.0) | 20% | 模拟轻微对焦不准 |
| Sharpness | factor=(0.5,1.5) | 30% | 模拟相机锐度差异 |

### Mixing 增强

| 增强 | 参数 | 概率 | 理由 |
|------|------|------|------|
| CutMix | α=0.5 | 50% | 混合不同等级图像, 增强类边界识别, 与 EDL 边界不确定性先验互补 |

## 3. 配置代码

```python
train_transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomApply([RandomRotation(degrees=10)], p=0.5),
    RandomApply([RandomResizedCrop(size=(256,512), scale=(0.8,1.0))], p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1,1.0))], p=0.2),
    RandomApply([RandomAdjustSharpness(sharpness_factor=(0.5, 1.5))], p=0.3),
    Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# CutMix 需在 batch 级别施加, 不在 per-image Compose 中
# 在 DataLoader 的 collate_fn 或 training loop 中调用:
# from torchvision.transforms.v2 import CutMix
# cutmix = CutMix(alpha=0.5, num_classes=4)
# images, labels = cutmix(images, labels)  # p=0.5 内部随机

val_transform = Compose([
    Resize((256,512)),
    Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
```

## 4. 禁用增强

- **垂直翻转**: 牙齿上下方向有解剖含义 (切缘 vs 牙龈)
- **大角度旋转 (>15°)**: 破坏牙弓曲线方向, 影响 Arch Scan 序列建模
- **高幅度色相偏移**: 可能改变白垩色的诊断信号
- **MixUp**: 与 CutMix 功能重叠, 选 CutMix (更直观的空间混合)
