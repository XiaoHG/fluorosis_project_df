#!/bin/bash
# =============================================================================
# push_training_data.sh — 推送训练监控数据到 GitHub
#
# 用法:
#   bash scripts/push_training_data.sh                          # 推送所有 logs
#   bash scripts/push_training_data.sh logs/e4_full_symmamba_fold0  # 推送指定实验
#
# 工作流:
#   1. 训练中你观察到异常 → Ctrl+C 停止训练
#   2. 运行本脚本: bash scripts/push_training_data.sh
#   3. 通知 Claude: "训练数据已推送, 请拉取分析"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_DIR"

TARGET="${1:-logs}"

echo "===== 推送训练数据 ====="
echo "目标: $TARGET"
echo "工作目录: $(pwd)"

# 检查目标是否存在
if [ ! -e "$TARGET" ]; then
    echo "错误: $TARGET 不存在"
    exit 1
fi

# 检查是否有变更
if git diff --quiet -- "$TARGET" && git diff --cached --quiet -- "$TARGET" && \
   [ -z "$(git ls-files --others --exclude-standard -- "$TARGET")" ]; then
    echo "没有新的训练数据需要推送"
    echo "当前 logs 状态:"
    find "$TARGET" -name "*.json" -o -name "*.jsonl" -o -name "*.csv" 2>/dev/null | head -20
    exit 0
fi

# 添加变更
echo ""
echo "添加文件..."
git add "$TARGET"

# 显示将要提交的内容
echo ""
echo "===== 变更文件 ====="
git diff --cached --stat -- "$TARGET"

# 提取实验名作为 commit message
EXP_NAMES=$(find "$TARGET" -name "run_info.json" -maxdepth 2 2>/dev/null | \
    sed 's|.*/logs/||;s|/run_info.json||' | tr '\n' ' ' || echo "unknown")
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')

git commit -m "training data: ${EXP_NAMES}@ ${TIMESTAMP}"

echo ""
echo "===== 推送到 GitHub ====="
git push origin main

echo ""
echo "Done! 训练数据已推送. 通知 Claude:"
echo "  '训练数据已推送, 实验: ${EXP_NAMES}, 请拉取分析'"
