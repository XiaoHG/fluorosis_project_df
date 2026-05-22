# SymMamba — 对称性感知双路径 Mamba 网络用于氟斑牙分级

## 项目结构

```
06_Implementation/
├── code/
│   ├── data/
│   │   ├── dataset.py          # FluorosisDataset, 5-fold split
│   │   └── augmentations.py    # 在线增强管线 + CutMix
│   ├── models/
│   │   ├── symmamba.py         # SymMamba 主模型 (~4M params)
│   │   ├── edl_head.py         # Dirichlet 证据分类头
│   │   ├── losses.py           # L_EDL + L_ord + L_cont + L_boundary
│   │   └── baselines.py        # B1 ResNet50, B2 ResNet50+CORAL, B3 ViT
│   ├── utils/
│   │   ├── metrics.py          # QWK, SDR, Macro F1, ECE, UR-Acc
│   │   └── logger.py           # CSV 日志 + checkpoint 管理
│   ├── train.py                # 训练脚本
│   └── evaluate.py             # 评估脚本
├── configs/
│   ├── default.yaml            # 默认配置 (含 6 个 device profiles)
│   ├── proposed.yaml           # 完整 SymMamba (E6)
│   └── baselines/              # B1/B2/B3 配置文件
└── requirements.txt
```

## 环境配置

### PyTorch 安装

服务器 GPU: NVIDIA RTX PRO 6000 Blackwell (96GB, CUDA 13.0)

```bash
# CUDA 13.0 / Blackwell 需要 PyTorch >= 2.6
pip install torch>=2.6.0 torchvision>=0.21.0 --index-url https://download.pytorch.org/whl/cu130
```

如遇 cu130 index 不可用, 使用 nightly:
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

### 其余依赖

```bash
pip install -r requirements.txt
```

## 数据集部署

### 服务器部署

```bash
# 在项目根目录下创建数据集目录结构
mkdir -p dataset_df/dental_fluorosis/images/{normal,mild,moderate,severe}

# 将 200 张 PNG 图像按类别放入对应文件夹:
#   normal/   → 50 张正常牙齿照片
#   mild/     → 50 张轻度氟斑牙照片
#   moderate/ → 50 张中度氟斑牙照片
#   severe/   → 50 张重度氟斑牙照片
```

### 验证部署

```bash
python -c "
from pathlib import Path
root = Path('dataset_df/dental_fluorosis/images')
for cls in ['normal','mild','moderate','severe']:
    n = len(list((root/cls).glob('*.png')))
    print(f'{cls}: {n} images')
"
```

预期输出每类 50 张, 总计 200 张。

数据集不包含在此仓库中 (见 `.gitignore`)。

## 训练

### 完整 SymMamba (E6)

```bash
cd 06_Implementation
python code/train.py --config configs/proposed.yaml --profile rtx_pro_6000
```

### Baseline 模型

```bash
# B1: ResNet50
python code/train.py --config configs/baselines/b1_resnet50.yaml

# B2: ResNet50 + CORAL
python code/train.py --config configs/baselines/b2_resnet50_coral.yaml

# B3: ViT-B/16 (MLTrMR)
python code/train.py --config configs/baselines/b3_vit_mltrmr.yaml
```

### 关键参数

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径 |
| `--profile` | 设备配置 (rtx_pro_6000 / rtx4090 / rtx3090 / rtx4070ti / cpu) |
| `--fold` | 单折训练 (-1 = 全部 5 折) |
| `--resume` | 从 checkpoint 恢复 |
| `--data_root` | 覆盖数据集路径 |
| `--exp_name` | 覆盖实验名称 |

### 设备配置切换

在 `configs/default.yaml` 中预定义了 6 个 device profiles, 通过 `--profile` 切换:

```bash
--profile rtx_pro_6000    # batch_size=16, amp=true,  workers=8
--profile rtx4090          # batch_size=16, amp=true,  workers=8
--profile rtx4070ti        # batch_size=12, amp=true,  workers=4
--profile cpu              # batch_size=2,  amp=false, workers=0
```

## 评估

```bash
python code/evaluate.py \
  --config configs/proposed.yaml \
  --checkpoint logs/proposed_full_fold0/checkpoints/best.pt \
  --output_dir evaluation_results
```

## 输出

训练每个 fold 生成:

```
logs/{exp_name}_fold{N}/
├── metrics.csv          # 每 epoch 的 train/val 指标
├── checkpoints/
│   └── best.pt          # 最佳模型权重
└── summary.json         # 最终评估汇总
```

## 模型架构

SymMamba: ~4M 参数, 3 阶段双路径 Mamba 网络

- **Arch Scan**: 沿牙弓水平方向序列建模
- **Cross Scan**: 左右半侧对称性感知 (共享权重)
- **CGF x3**: Cross Gated Fusion 逐阶段融合双路径特征
- **EDL Head**: Dirichlet 证据输出 (alpha, belief, u, pred)
- **损失**: L_EDL + 0.1 L_ord + 0.05 L_cont + 0.01 L_boundary
