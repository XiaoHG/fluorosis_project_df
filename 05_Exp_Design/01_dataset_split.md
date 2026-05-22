---
date: 2026-05-21
author: ExpDesignAgent
input_from: "01_Knowledge_Base/data_description.md, 04_Model_Design/02_selected_architecture.md"
output_to: "05_Exp_Design/01_dataset_split.md"
status: draft
---

# 数据集划分方案

## 1. 数据分布

| 项目 | 值 |
|------|-----|
| 总样本 | 200 (4 类 × 50, 完全平衡) |
| 图像尺寸 | 512×256 (W×H, 2:1) |
| 格式 | PNG |
| 标注 | 临床医生 Dean 指数 (0-3) |
| 标注一致性 | 未量化 (已知牙医间 <70%, 建议补充) |

## 2. 划分策略: 5 折分层交叉验证

选择理由: 200 张样本量小, 固定 hold-out 验证+测试仅 80 张, 统计效力不足。5-fold CV 每折 160 train / 40 val, 所有样本均被验证一次。

| 参数 | 值 |
|------|-----|
| 折数 | 5 |
| 每折 train | 160 (4 类×40) |
| 每折 val | 40 (4 类×10) |
| 方法 | sklearn StratifiedKFold |
| 种子 | 42 |
| 划分索引 | 保存至 `split_indices.json` |

## 3. 测试策略

5-fold CV 的每折验证集 (=40) 同时承担验证和测试功能:
- 模型选择 (early stop, 超参) 基于 val QWK
- 最终指标为 5 折汇总 (均值 ± std)
- 论文中说明 "no independent test set due to limited sample size"

## 4. 防泄露

| 风险 | 措施 |
|------|------|
| 同患者跨折 | 假设每图像 = 独立患者, 否则按 subject 分组 |
| 增强泄露 | 仅 train 增强, val 仅归一化 |
| 信息泄露 | 所有超参选择 (含 θ 校准) 折内完成 |
| SSL 预训练泄露 | COde/Oral Diseases 与氟斑牙无重叠 |

## 5. 复现性

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

划分索引首次运行后固定为 JSON, 所有实验 (模型/对比/消融) 使用同一划分。

## 6. [待确认]

数据集中是否存在同一患者的多张图像? 如存在, 需按患者 ID 分组划分。
