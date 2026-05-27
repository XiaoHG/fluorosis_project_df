#!/bin/bash
# 每日项目整理脚本 — 检查 + 生成报告 + 发送邮件
# 用法: ./daily_organize.sh
# cron: 3 21 * * * cd /path/to/project && ./00_Admin/scripts/daily_organize.sh

set -euo pipefail

PROJECT_DIR="/Volumes/KINGSTON/fluorosis_project/fluorosis_project_df"
cd "$PROJECT_DIR"

REPORT_FILE=$(mktemp)
trap "rm -f $REPORT_FILE" EXIT

{
    echo "============================================"
    echo "  氟斑牙项目整理报告 — $(date '+%Y-%m-%d %H:%M')"
    echo "============================================"
    echo ""

    # 1. Git 状态
    echo "## 1. Git 状态"
    echo "----------------------------------------"
    git status --short 2>&1 || echo "  [错误] 无法获取 git status"
    echo ""

    # 2. 双扩展名检查
    echo "## 2. .md.md 双扩展名检查"
    echo "----------------------------------------"
    DOUBLE_EXT=$(find . -name "*.md.md" -not -path "./.git/*" -not -path "./.claude/*" -not -path "./dataset_df/*" 2>/dev/null)
    if [ -z "$DOUBLE_EXT" ]; then
        echo "  [OK] 未发现双扩展名文件"
    else
        echo "$DOUBLE_EXT"
        echo "  [修复] 自动修复中..."
        for f in $DOUBLE_EXT; do
            mv "$f" "${f%.md.md}.md"
            echo "    已修复: $f -> ${f%.md.md}.md"
        done
    fi
    echo ""

    # 3. .gitignore 检查
    echo "## 3. .gitignore 完整性"
    echo "----------------------------------------"
    if grep -q ".obsidian/" .gitignore 2>/dev/null; then
        echo "  [OK] .obsidian/ 已忽略"
    else
        echo "  [警告] .obsidian/ 未在 .gitignore 中"
    fi
    if grep -q "dataset_df/" .gitignore 2>/dev/null; then
        echo "  [OK] dataset_df/ 已忽略"
    else
        echo "  [警告] dataset_df/ 未在 .gitignore 中"
    fi
    echo ""

    # 4. 大文件检查 (>10MB)
    echo "## 4. 大文件检查 (>10MB)"
    echo "----------------------------------------"
    LARGE=$(find . -type f -size +10M -not -path "./.git/*" -not -path "./dataset_df/*" 2>/dev/null | head -10)
    if [ -z "$LARGE" ]; then
        echo "  [OK] 未发现大文件"
    else
        echo "$LARGE"
        echo "  [警告] 以上文件超过 10MB，确认是否需要加入 .gitignore 或使用 Git LFS"
    fi
    echo ""

    # 5. 临时文件检查
    echo "## 5. 临时文件检查"
    echo "----------------------------------------"
    TMP_FILES=$(find . -name "*.tmp" -o -name "*.pyc" -o -name ".DS_Store" -not -path "./.git/*" 2>/dev/null | head -10)
    if [ -z "$TMP_FILES" ]; then
        echo "  [OK] 未发现临时文件"
    else
        echo "$TMP_FILES"
        echo "  [警告] 存在临时文件"
    fi
    echo ""

    # 6. 文档目录新文件
    echo "## 6. 文档目录新文件 (未跟踪)"
    echo "----------------------------------------"
    for d in 00_Admin 01_Knowledge_Base 02_Literature 03_Innovation 04_Model_Design 05_Exp_Design 09_Review 10_Submission 11_Decision_Logs; do
        if [ -d "$d" ]; then
            UNTRACKED=$(git ls-files --others --exclude-standard "$d/" 2>/dev/null)
            if [ -n "$UNTRACKED" ]; then
                echo "  [$d]"
                echo "$UNTRACKED" | sed 's/^/    /'
            fi
        fi
    done
    echo "  (无新文件则无输出)"
    echo ""

    # 7. 健康度简评
    echo "## 7. 项目结构健康度"
    echo "----------------------------------------"
    ISSUES=0
    if [ -n "${DOUBLE_EXT:-}" ]; then ISSUES=$((ISSUES + 1)); fi
    if [ -n "${LARGE:-}" ]; then ISSUES=$((ISSUES + 1)); fi
    if [ -n "${TMP_FILES:-}" ]; then ISSUES=$((ISSUES + 1)); fi

    if [ $ISSUES -eq 0 ]; then
        echo "  [健康] 项目结构良好，无需处理"
    else
        echo "  [注意] 发现 $ISSUES 类问题，已自动修复可修复项"
    fi
    echo ""
    echo "============================================"
    echo "  报告结束"
    echo "============================================"

} > "$REPORT_FILE"

cat "$REPORT_FILE"

if [ -n "${GMAIL_APP_PASSWORD:-}" ]; then
    echo ""
    echo ">>> 发送报告到 xiaohggg@gmail.com ..."
    cat "$REPORT_FILE" | python3 "$PROJECT_DIR/00_Admin/scripts/send_report.py"
else
    echo ""
    echo ">>> 未设置 GMAIL_APP_PASSWORD，跳过邮件发送"
    echo ">>> 设置方法: export GMAIL_APP_PASSWORD='your-app-password'"
fi
