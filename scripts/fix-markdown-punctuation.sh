#!/bin/bash

# Markdown中文标点修复脚本 (Shell版本)
# 
# 使用方法:
#   ./scripts/fix-markdown-punctuation.sh [目录]
#   ./scripts/fix-markdown-punctuation.sh docs/
#
# 功能:
# - 修复中文冒号、括号、引号
# - 为粗体标签后添加空格
# - 自动备份原文件

set -e

# 默认目录
TARGET_DIR="${1:-docs/}"

# 检查目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 目录 $TARGET_DIR 不存在"
    exit 1
fi

echo "Markdown中文标点修复工具 (Shell版本)"
echo "目标目录: $(realpath "$TARGET_DIR")"
echo "============================================"

# 查找所有markdown文件
MD_FILES=$(find "$TARGET_DIR" -name "*.md" -type f)

if [ -z "$MD_FILES" ]; then
    echo "在 $TARGET_DIR 中未找到markdown文件"
    exit 0
fi

# 计数器
TOTAL_FILES=0
MODIFIED_FILES=0

# 处理每个文件
while IFS= read -r file; do
    if [ -z "$file" ]; then
        continue
    fi
    
    TOTAL_FILES=$((TOTAL_FILES + 1))
    echo ""
    echo "处理: $(realpath --relative-to="$TARGET_DIR" "$file")"
    
    # 创建临时文件
    TEMP_FILE=$(mktemp)
    BACKUP_FILE="${file}.bak"
    
    # 标记是否有修改
    MODIFIED=false
    
    # 复制原文件到临时文件
    cp "$file" "$TEMP_FILE"
    
    # 修复中文标点
    if grep -q '[：（）""'']' "$TEMP_FILE"; then
        sed -i '' 's/：/:/g; s/（/(/g; s/）/)/g; s/"/"/g; s/"/"/g; s/'\''/'"'"'/g; s/'\''/'"'"'/g' "$TEMP_FILE"
        MODIFIED=true
        echo "  ✓ 修复中文标点"
    fi
    
    # 修复粗体标签后缺少空格的问题
    if grep -q '\*\*[^*]*:\*\*[A-Za-z\u4e00-\u9fff]' "$TEMP_FILE"; then
        # 使用更复杂的sed命令来添加空格
        sed -i '' 's/\(\*\*[^*]*:\*\*\)\([A-Za-z一-龯]\)/\1 \2/g' "$TEMP_FILE"
        MODIFIED=true
        echo "  ✓ 为粗体标签后添加空格"
    fi
    
    # 如果有修改，则更新文件
    if [ "$MODIFIED" = true ]; then
        # 创建临时备份
        cp "$file" "$BACKUP_FILE"
        
        # 应用修改
        if cp "$TEMP_FILE" "$file"; then
            # 修改成功，删除备份文件
            rm -f "$BACKUP_FILE"
            MODIFIED_FILES=$((MODIFIED_FILES + 1))
            echo "  ✓ 文件已修复"
        else
            # 修改失败，恢复备份
            echo "  ✗ 文件修改失败，恢复备份"
            cp "$BACKUP_FILE" "$file"
            rm -f "$BACKUP_FILE"
        fi
    else
        echo "  ✓ 无需修改"
    fi
    
    # 清理临时文件
    rm -f "$TEMP_FILE"
    
done <<< "$MD_FILES"

echo ""
echo "============================================"
echo "修复完成: $MODIFIED_FILES/$TOTAL_FILES 个文件被修改"

if [ $MODIFIED_FILES -gt 0 ]; then
    echo ""
    echo "所有文件修改已完成，未保留备份文件"
fi 