#!/usr/bin/env python3
"""
MDX表格错误修复脚本

专门解决Docusaurus MDX编译时表格中数字开头内容导致的错误，
如: "Unexpected character `1` (U+0031) before name"

常见问题:
- 表格单元格中的数字范围 (如: 1-4, 1.5-2x)
- 百分比数字 (如: 0.1%, 50%)
- 版本号 (如: 2.0, 3.14)

解决方案:
- 使用反引号包装数字开头的内容
- 保持表格结构不变
- 智能识别需要包装的内容

使用方法:
python scripts/fix-mdx-table-errors.py [目录路径] [选项]
"""

import os
import re
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict


class MDXTableFixer:
    def __init__(self):
        # 匹配表格行的模式
        self.table_row_pattern = re.compile(r'^\s*\|.*\|\s*$')
        
        # 匹配需要用反引号包装的内容模式
        self.needs_wrapping_patterns = [
            # 数字开头的范围 (如: 1-4, 1.5-2x, 32+)
            re.compile(r'(\d+(?:\.\d+)?[-+]\w*)'),
            # 小于号开头的数字 (如: <10ms, <0.1%)
            re.compile(r'(<\d+(?:\.\d+)?[%\w]*)'),
            # 大于号开头的数字 (如: >100ms, >2%)
            re.compile(r'(>\d+(?:\.\d+)?[%\w]*)'),
            # 纯百分比数字 (如: 50%, 0.1%)
            re.compile(r'(\d+(?:\.\d+)?%)'),
            # 版本号格式 (如: 1.5, 2.0)
            re.compile(r'(\d+\.\d+(?:x)?)'),
        ]
    
    def is_table_row(self, line: str) -> bool:
        """判断是否为表格行"""
        return bool(self.table_row_pattern.match(line))
    
    def fix_table_cell(self, cell_content: str) -> str:
        """修复单个表格单元格的内容"""
        cell_content = cell_content.strip()
        
        # 如果内容已经被反引号包装，跳过
        if cell_content.startswith('`') and cell_content.endswith('`'):
            return cell_content
        
        # 检查是否需要包装
        for pattern in self.needs_wrapping_patterns:
            if pattern.fullmatch(cell_content):
                return f'`{cell_content}`'
        
        return cell_content
    
    def fix_table_row(self, line: str) -> str:
        """修复表格行"""
        if not self.is_table_row(line):
            return line
        
        # 分割表格行
        parts = line.split('|')
        
        # 修复每个单元格（跳过第一个和最后一个空部分）
        fixed_parts = []
        for i, part in enumerate(parts):
            if i == 0 or i == len(parts) - 1:
                # 保持首尾的空格和格式
                fixed_parts.append(part)
            else:
                # 修复中间的单元格内容
                fixed_content = self.fix_table_cell(part)
                # 保持原有的前后空格
                leading_spaces = len(part) - len(part.lstrip())
                trailing_spaces = len(part) - len(part.rstrip())
                if leading_spaces > 0 or trailing_spaces > 0:
                    fixed_parts.append(' ' * leading_spaces + fixed_content.strip() + ' ' * trailing_spaces)
                else:
                    fixed_parts.append(fixed_content)
        
        return '|'.join(fixed_parts)
    
    def scan_file(self, file_path: Path) -> Dict[str, List[Tuple[int, str]]]:
        """扫描文件，返回发现的问题"""
        issues = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"无法读取文件 {file_path}: {e}")
            return issues
        
        problem_lines = []
        for line_num, line in enumerate(lines, 1):
            if self.is_table_row(line):
                # 检查表格行中是否有需要修复的内容
                parts = line.split('|')
                for part in parts[1:-1]:  # 跳过首尾空部分
                    cell_content = part.strip()
                    for pattern in self.needs_wrapping_patterns:
                        if pattern.fullmatch(cell_content) and not (cell_content.startswith('`') and cell_content.endswith('`')):
                            problem_lines.append((line_num, line.strip()))
                            break
        
        if problem_lines:
            issues['MDX表格数字格式问题'] = problem_lines
        
        return issues
    
    def fix_content(self, content: str) -> Tuple[str, List[str]]:
        """修复内容，返回修复后的内容和修改记录"""
        changes = []
        lines = content.split('\n')
        fixed_lines = []
        
        for line_num, line in enumerate(lines):
            if self.is_table_row(line):
                fixed_line = self.fix_table_row(line)
                if fixed_line != line:
                    changes.append(f"第{line_num + 1}行: 修复表格数字格式")
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines), changes
    
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
                    print(f"      原文: {orig}")
                    print(f"      修改: {fixed}")
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
            
            # 扫描问题
            issues = self.scan_file(file_path)
            
            if not issues:
                print("  ✓ 无问题")
                continue
            
            # 统计问题
            file_issues = sum(len(issue_list) for issue_list in issues.values())
            total_issues += file_issues
            
            # 显示问题
            for issue_type, issue_list in issues.items():
                print(f"  发现 {len(issue_list)} 个 {issue_type}:")
                for line_num, line_content in issue_list[:3]:  # 只显示前3个
                    print(f"    第{line_num}行: {line_content[:60]}{'...' if len(line_content) > 60 else ''}")
                if len(issue_list) > 3:
                    print(f"    ... 还有 {len(issue_list) - 3} 个问题")
            
            # 处理文件
            if check_only:
                continue
            elif preview:
                self.preview_changes(file_path)
            else:
                if self.fix_file(file_path, backup):
                    total_fixed += 1
                    print(f"  ✓ 已修复")
                else:
                    print(f"  ✓ 无需修改")
        
        # 总结
        print("\n" + "=" * 50)
        if check_only:
            print(f"检查完成: 共发现 {total_issues} 个问题")
        elif preview:
            print(f"预览完成: 共 {total_issues} 个问题等待修复")
        else:
            print(f"修复完成: {total_fixed} 个文件被修改")


def main():
    parser = argparse.ArgumentParser(description='MDX表格错误修复工具')
    parser.add_argument('directory', nargs='?', default='docs/', 
                       help='要处理的目录路径 (默认: docs/)')
    parser.add_argument('--check-only', action='store_true',
                       help='只检查问题，不修改文件')
    parser.add_argument('--preview', action='store_true',
                       help='预览修改内容')
    parser.add_argument('--backup', action='store_true',
                       help='修改前创建备份文件')
    
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.exists():
        print(f"错误: 目录 {directory} 不存在")
        return 1
    
    print("MDX表格错误修复工具")
    print(f"目标目录: {directory.absolute()}")
    if args.check_only:
        print("模式: 检查模式")
    elif args.preview:
        print("模式: 预览模式")
    else:
        print("模式: 修复模式")
    print("=" * 50)
    
    fixer = MDXTableFixer()
    fixer.process_directory(directory, args.check_only, args.preview, args.backup)
    
    return 0


if __name__ == '__main__':
    exit(main()) 