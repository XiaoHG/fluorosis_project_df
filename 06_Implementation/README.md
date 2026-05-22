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
│   ├── proposed.yaml           # SymMamba E4 (核心消融, 不含 SSL)
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

### SymMamba E4 (核心消融: Arch+Cross+CGFx3+Ord EDL, 无 SSL)

```bash
cd 06_Implementation
python code/train.py --config configs/proposed.yaml --profile rtx_pro_6000
```

> **E5/E6 (含 Oral SSL 预训练) 见下方实验路线图。** E4 先验证核心架构, SSL 预训练权重就绪后加载。

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
  --checkpoint logs/e4_full_symmamba_fold0/checkpoints/best.pt \
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

## 实验路线图

按消融计划 (`05_Exp_Design/05_ablation_plan.md`) 的实验顺序:

| 阶段 | 实验 | 描述 | 状态 |
|------|------|------|------|
| P0 | E0 | ResNet50 baseline | 代码就绪 |
| P0 | **E4** | **Full SymMamba + Ord EDL (无 SSL)** | **当前训练** |
| P1 | E3 | Arch Scan only (消融 Cross Scan) | 待 E4 完成 |
| P1 | E5 | E4 + Oral SSL 预训练加载 | 待公开数据集下载 + DINOv2 预训练 |
| P2 | E6 | 完整方案 (= E5, 论文 Ours) | 待 E5 完成 |

### E5/E6: Oral SSL 预训练 (待实现)

需下载 3 个公开口腔数据集 (~65K 张) 做 DINOv2 自监督预训练:

| 数据集 | 规模 | 用途 |
|--------|------|------|
| COde (Caries Ontology) | ~50K | 龋齿/口腔病变 |
| Oral Diseases | ~13K | 口腔疾病分类 |
| AlphaDent | ~1.3K | 牙齿分割 |

预训练完成后, config 中启用:
```yaml
model:
  ssl:
    enabled: true
    checkpoint: "path/to/dinov2_oral_pretrained.pt"
```
