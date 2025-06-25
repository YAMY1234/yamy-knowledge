#!/usr/bin/env python3
"""
Markdown中文标点修复脚本

功能:
1. 检查markdown文件中的中文标点问题
2. 自动修复中文冒号、括号等标点符号
3. 修复**标签:**后面缺少空格的格式问题
4. 支持批量处理和预览模式

使用方法:
python scripts/fix-markdown-punctuation.py [目录路径] [选项]

选项:
--check-only: 只检查问题，不修改文件
--preview: 预览修改内容
--backup: 修改前创建备份文件
"""

import os
import re
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict


class MarkdownPunctuationFixer:
    def __init__(self):
        # 需要修复的中文标点映射
        self.punctuation_fixes = {
            '：': ':',    # 中文冒号 -> 英文冒号
            '（': '(',    # 中文左括号 -> 英文左括号
            '）': ')',    # 中文右括号 -> 英文右括号
            '"': '"',     # 中文左引号 -> 英文引号
            '"': '"',     # 中文右引号 -> 英文引号
            ''': "'",     # 中文左单引号 -> 英文单引号
            ''': "'",     # 中文右单引号 -> 英文单引号
        }
        
        # 需要在冒号后添加空格的粗体标签模式
        self.bold_label_pattern = re.compile(r'\*\*([^*]+):\*\*(?!\s)')
        
        # 检查中文标点的模式
        self.chinese_punct_patterns = {
            '中文冒号': re.compile(r'[：]'),
            '中文括号': re.compile(r'[（）]'),
            # 临时注释掉中文引号检测，因为现在文件中只有标准英文引号
            # '中文引号': re.compile(r'[""'']'),
        }
        
        # 检查格式问题的模式
        self.format_patterns = {
            '粗体标签后缺少空格': re.compile(r'\*\*[^*]+:\*\*[A-Za-z\u4e00-\u9fff]'),
        }

    def scan_file(self, file_path: Path) -> Dict[str, List[Tuple[int, str]]]:
        """扫描文件，返回发现的问题"""
        issues = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"无法读取文件 {file_path}: {e}")
            return issues
        
        # 跟踪代码块状态
        in_code_block = False
        
        # 检查中文标点问题
        for category, pattern in self.chinese_punct_patterns.items():
            found_issues = []
            in_code_block = False  # 重置代码块状态
            
            for line_num, line in enumerate(lines, 1):
                # 检查代码块边界
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                
                # 跳过代码块内的内容
                if in_code_block:
                    continue
                    
                matches = pattern.findall(line)
                if matches:
                    found_issues.append((line_num, line.strip()))
            if found_issues:
                issues[category] = found_issues
        
        # 检查格式问题
        for category, pattern in self.format_patterns.items():
            found_issues = []
            in_code_block = False  # 重置代码块状态
            
            for line_num, line in enumerate(lines, 1):
                # 检查代码块边界
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                
                # 跳过代码块内的内容
                if in_code_block:
                    continue
                    
                if pattern.search(line):
                    found_issues.append((line_num, line.strip()))
            if found_issues:
                issues[category] = found_issues
        
        return issues

    def fix_content(self, content: str) -> Tuple[str, List[str]]:
        """修复内容，返回修复后的内容和修改记录"""
        changes = []
        lines = content.split('\n')
        
        # 跟踪代码块状态
        in_code_block = False
        
        # 逐行处理，排除代码块内容
        for i, line in enumerate(lines):
            # 检查代码块边界
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                # 不要continue，要继续处理当前行（但不修复代码块标记行）
            
            # 跳过代码块内的内容
            if in_code_block:
                continue
            
            # 修复中文标点
            original_line = line
            for chinese_punct, english_punct in self.punctuation_fixes.items():
                if chinese_punct in line:
                    line = line.replace(chinese_punct, english_punct)
            
            # 修复粗体标签后缺少空格的问题
            def add_space_after_colon(match):
                return match.group(0) + ' '
            
            line = self.bold_label_pattern.sub(add_space_after_colon, line)
            
            # 更新行内容
            if line != original_line:
                lines[i] = line
        
        fixed_content = '\n'.join(lines)
        
        # 统计修改次数
        for chinese_punct, english_punct in self.punctuation_fixes.items():
            original_count = content.count(chinese_punct)
            fixed_count = fixed_content.count(chinese_punct)
            if original_count > fixed_count:
                changes.append(f"替换 {original_count - fixed_count} 个 '{chinese_punct}' -> '{english_punct}'")
        
        # 统计粗体标签修复
        original_bold_issues = len(self.bold_label_pattern.findall(content))
        fixed_bold_issues = len(self.bold_label_pattern.findall(fixed_content))
        if original_bold_issues > fixed_bold_issues:
            changes.append(f"为 {original_bold_issues - fixed_bold_issues} 个粗体标签后添加空格")
        
        return fixed_content, changes

    def fix_file(self, file_path: Path, backup: bool = False) -> bool:
        """修复单个文件"""
        try:
            # 读取原文件
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # 修复内容
            fixed_content, changes = self.fix_content(original_content)
            
            # 如果没有变化，跳过
            if fixed_content == original_content:
                return False
            
            # 创建备份
            if backup:
                backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                shutil.copy2(file_path, backup_path)
                print(f"  备份文件: {backup_path}")
            
            # 写入修复后的内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            # 打印修改记录
            for change in changes:
                print(f"  {change}")
            
            return True
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return False

    def preview_changes(self, file_path: Path) -> None:
        """预览文件的修改内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            fixed_content, changes = self.fix_content(original_content)
            
            if fixed_content == original_content:
                print(f"  无需修改")
                return
            
            print(f"  预计修改:")
            for change in changes:
                print(f"    - {change}")
            
            # 显示部分差异示例
            original_lines = original_content.split('\n')
            fixed_lines = fixed_content.split('\n')
            
            diff_count = 0
            for i, (orig, fixed) in enumerate(zip(original_lines, fixed_lines)):
                if orig != fixed and diff_count < 3:  # 只显示前3个差异
                    print(f"    第{i+1}行:")
                    print(f"      原文: {orig[:80]}{'...' if len(orig) > 80 else ''}")
                    print(f"      修改: {fixed[:80]}{'...' if len(fixed) > 80 else ''}")
                    diff_count += 1
            
            if diff_count == 3 and len([1 for o, f in zip(original_lines, fixed_lines) if o != f]) > 3:
                print(f"    ... 还有更多修改")
                
        except Exception as e:
            print(f"预览文件 {file_path} 时出错: {e}")

    def process_directory(self, directory: Path, check_only: bool = False, 
                         preview: bool = False, backup: bool = False) -> None:
        """处理目录中的所有markdown文件"""
        md_files = list(directory.rglob('*.md'))
        
        if not md_files:
            print(f"在 {directory} 中未找到markdown文件")
            return
        
        print(f"找到 {len(md_files)} 个markdown文件")
        print("-" * 50)
        
        total_issues = 0
        total_fixed = 0
        
        for file_path in md_files:
            relative_path = file_path.relative_to(directory)
            print(f"\n处理: {relative_path}")
            
            if check_only:
                # 仅检查模式
                issues = self.scan_file(file_path)
                if issues:
                    total_issues += 1
                    print(f"  发现问题:")
                    for category, issue_list in issues.items():
                        print(f"    {category}: {len(issue_list)} 处")
                        for line_num, line in issue_list[:3]:  # 只显示前3个
                            print(f"      第{line_num}行: {line[:60]}{'...' if len(line) > 60 else ''}")
                        if len(issue_list) > 3:
                            print(f"      ... 还有 {len(issue_list) - 3} 处")
                else:
                    print(f"  ✓ 无问题")
            
            elif preview:
                # 预览模式
                self.preview_changes(file_path)
            
            else:
                # 修复模式
                if self.fix_file(file_path, backup):
                    total_fixed += 1
                    print(f"  ✓ 已修复")
                else:
                    print(f"  ✓ 无需修改")
        
        # 总结
        print("\n" + "=" * 50)
        if check_only:
            print(f"检查完成: {total_issues}/{len(md_files)} 个文件有问题")
        elif preview:
            print(f"预览完成: 共检查 {len(md_files)} 个文件")
        else:
            print(f"修复完成: {total_fixed}/{len(md_files)} 个文件被修改")


def main():
    parser = argparse.ArgumentParser(
        description='修复markdown文件中的中文标点问题',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python scripts/fix-markdown-punctuation.py docs/               # 修复docs目录下的所有md文件
  python scripts/fix-markdown-punctuation.py docs/ --check-only  # 仅检查问题
  python scripts/fix-markdown-punctuation.py docs/ --preview     # 预览修改
  python scripts/fix-markdown-punctuation.py docs/ --backup      # 修复前创建备份
        """
    )
    
    parser.add_argument('directory', nargs='?', default='docs/',
                       help='要处理的目录路径 (默认: docs/)')
    parser.add_argument('--check-only', action='store_true',
                       help='仅检查问题，不修改文件')
    parser.add_argument('--preview', action='store_true',
                       help='预览修改内容，不实际修改')
    parser.add_argument('--backup', action='store_true',
                       help='修改前创建备份文件')
    
    args = parser.parse_args()
    
    # 验证目录
    directory = Path(args.directory)
    if not directory.exists():
        print(f"错误: 目录 {directory} 不存在")
        return
    
    if not directory.is_dir():
        print(f"错误: {directory} 不是一个目录")
        return
    
    # 创建修复器并处理
    fixer = MarkdownPunctuationFixer()
    
    print(f"Markdown中文标点修复工具")
    print(f"目标目录: {directory.absolute()}")
    
    if args.check_only:
        print("模式: 仅检查问题")
    elif args.preview:
        print("模式: 预览修改")
    else:
        print("模式: 修复文件" + (" (含备份)" if args.backup else ""))
    
    fixer.process_directory(directory, args.check_only, args.preview, args.backup)


if __name__ == '__main__':
    main() 