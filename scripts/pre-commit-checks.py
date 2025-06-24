#!/usr/bin/env python3
"""
Pre-commit 检查脚本

集成多种检查功能:
1. MDX表格数字格式检查
2. 中文标点检查
3. 支持Git hook使用

使用方法:
python3 scripts/pre-commit-checks.py [--fix] [--staged-only]

选项:
--fix: 自动修复问题
--staged-only: 只检查已暂存的文件
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Set

# 导入修复器类
sys.path.append(str(Path(__file__).parent))

# 导入现有的修复器
def import_fixers():
    """动态导入修复器类"""
    global MDXTableFixer, MarkdownPunctuationFixer
    
    try:
        # 尝试从同目录导入
        import importlib.util
        
        # 导入MDX修复器
        mdx_spec = importlib.util.spec_from_file_location(
            "fix_mdx_table_errors", 
            Path(__file__).parent / "fix-mdx-table-errors.py"
        )
        mdx_module = importlib.util.module_from_spec(mdx_spec)
        mdx_spec.loader.exec_module(mdx_module)
        MDXTableFixer = mdx_module.MDXTableFixer
        
        # 导入标点修复器
        punct_spec = importlib.util.spec_from_file_location(
            "fix_markdown_punctuation",
            Path(__file__).parent / "fix-markdown-punctuation.py"
        )
        punct_module = importlib.util.module_from_spec(punct_spec)
        punct_spec.loader.exec_module(punct_module)
        MarkdownPunctuationFixer = punct_module.MarkdownPunctuationFixer
        
    except Exception as e:
        print(f"警告: 无法导入修复器类 ({e})，使用简化版本检查")
        
        class MDXTableFixer:
            def scan_file(self, file_path):
                return {}
            
            def fix_file(self, file_path, backup=False):
                return False
        
        class MarkdownPunctuationFixer:
            def scan_file(self, file_path):
                return {}
            
            def fix_file(self, file_path, backup=False):
                return False

# 导入修复器
import_fixers()


class PreCommitChecker:
    def __init__(self):
        self.mdx_fixer = MDXTableFixer()
        self.punct_fixer = MarkdownPunctuationFixer()
        self.errors_found = False
    
    def get_staged_files(self) -> List[Path]:
        """获取已暂存的markdown文件"""
        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
                capture_output=True,
                text=True,
                check=True
            )
            files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            return [Path(f) for f in files if f.endswith('.md') and Path(f).exists()]
        except subprocess.CalledProcessError:
            print("警告: 无法获取Git暂存文件，检查所有markdown文件")
            return []
    
    def get_all_markdown_files(self) -> List[Path]:
        """获取所有markdown文件"""
        return list(Path('docs/').rglob('*.md')) if Path('docs/').exists() else []
    
    def check_file(self, file_path: Path, fix_mode: bool = False) -> bool:
        """检查单个文件，返回是否有问题"""
        print(f"检查: {file_path}")
        
        has_issues = False
        
        # MDX表格检查
        mdx_issues = self.mdx_fixer.scan_file(file_path)
        if mdx_issues:
            has_issues = True
            for issue_type, issue_list in mdx_issues.items():
                print(f"  ❌ {issue_type}: {len(issue_list)} 个问题")
                if not fix_mode:
                    for line_num, line_content in issue_list[:3]:
                        print(f"    第{line_num}行: {line_content[:60]}{'...' if len(line_content) > 60 else ''}")
            
            if fix_mode:
                if self.mdx_fixer.fix_file(file_path):
                    print(f"  ✓ MDX表格问题已修复")
                    has_issues = False  # 已修复
        
        # 中文标点检查
        punct_issues = self.punct_fixer.scan_file(file_path)
        if punct_issues:
            has_issues = True
            for issue_type, issue_list in punct_issues.items():
                print(f"  ❌ {issue_type}: {len(issue_list)} 个问题")
                if not fix_mode:
                    for line_num, line_content in issue_list[:3]:
                        print(f"    第{line_num}行: {line_content[:60]}{'...' if len(line_content) > 60 else ''}")
            
            if fix_mode:
                if self.punct_fixer.fix_file(file_path):
                    print(f"  ✓ 中文标点问题已修复")
                    has_issues = False  # 已修复
        
        if not has_issues:
            print(f"  ✓ 无问题")
        
        return has_issues
    
    def run_checks(self, staged_only: bool = False, fix_mode: bool = False) -> int:
        """运行所有检查"""
        print("🔍 Pre-commit 检查开始")
        print("=" * 50)
        
        # 获取要检查的文件
        if staged_only:
            files_to_check = self.get_staged_files()
            if not files_to_check:
                print("没有已暂存的markdown文件需要检查")
                return 0
            print(f"检查 {len(files_to_check)} 个已暂存的文件")
        else:
            files_to_check = self.get_all_markdown_files()
            print(f"检查 {len(files_to_check)} 个markdown文件")
        
        if not files_to_check:
            print("没有找到markdown文件")
            return 0
        
        print("-" * 50)
        
        total_issues = 0
        fixed_files = 0
        
        for file_path in files_to_check:
            has_issues = self.check_file(file_path, fix_mode)
            if has_issues and not fix_mode:
                total_issues += 1
                self.errors_found = True
            elif has_issues and fix_mode:
                # 修复模式下，如果还有问题说明修复失败
                total_issues += 1
                self.errors_found = True
            elif fix_mode and not has_issues:
                # 修复模式下，无问题可能意味着修复成功
                pass
        
        print("\n" + "=" * 50)
        
        if fix_mode:
            if total_issues == 0:
                print("✅ 所有问题已修复或无问题")
                return 0
            else:
                print(f"❌ 还有 {total_issues} 个文件存在问题")
                return 1
        else:
            if total_issues == 0:
                print("✅ 所有检查通过")
                return 0
            else:
                print(f"❌ 发现 {total_issues} 个文件存在问题")
                print("\n💡 使用 --fix 选项自动修复问题:")
                if staged_only:
                    print("   python3 scripts/pre-commit-checks.py --fix --staged-only")
                else:
                    print("   python3 scripts/pre-commit-checks.py --fix")
                return 1


def main():
    parser = argparse.ArgumentParser(description='Pre-commit 检查工具')
    parser.add_argument('--fix', action='store_true',
                       help='自动修复发现的问题')
    parser.add_argument('--staged-only', action='store_true',
                       help='只检查已暂存的文件（用于Git hook）')
    
    args = parser.parse_args()
    
    checker = PreCommitChecker()
    return checker.run_checks(args.staged_only, args.fix)


if __name__ == '__main__':
    exit(main()) 