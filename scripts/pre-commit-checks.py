#!/usr/bin/env python3
"""
Pre-commit 检查脚本

集成多种检查功能:
1. MDX表格数字格式检查
2. 中文标点检查
3. MDX语法特殊字符检查
4. Markdown标题结构检查
5. 支持Git hook使用

使用方法:
python3 scripts/pre-commit-checks.py [--fix] [--staged-only]

选项:
--fix: 自动修复问题
--staged-only: 只检查已暂存的文件

标题结构检查功能:
- 检测长文档缺少二级标题层级结构的问题
- 检测和修复错误的粗体转义格式 (\*\*文本\*\*)
- 自动将常见的粗体文本模式转换为合适的标题格式
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


class MDXSyntaxChecker:
    """MDX语法检查器，检测可能导致MDX编译错误的特殊字符"""
    
    def __init__(self):
        # MDX中需要注意的特殊字符模式
        self.problematic_patterns = [
            (r'\|\s*[^|]*?\d+\+[^|]*?\|', '表格中的加号需要转义'),
            (r'\|\s*[^|]*?\d+\*[^|]*?\|', '表格中的星号需要转义'),
            (r'\|\s*[^|]*?[<>][^|]*?\|', '表格中的尖括号可能需要转义'),
            (r'\|\s*[^|]*?\{[^|}]*?\}[^|]*?\|', '表格中的花括号可能需要转义'),
        ]
    
    def scan_file(self, file_path):
        """扫描文件中的MDX语法问题"""
        import re
        issues = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                for pattern, description in self.problematic_patterns:
                    if re.search(pattern, line):
                        if description not in issues:
                            issues[description] = []
                        issues[description].append((line_num, line.strip()))
        
        except Exception as e:
            print(f"警告: 无法扫描文件 {file_path} ({e})")
        
        return issues
    
    def fix_file(self, file_path, backup=False):
        """修复文件中的MDX语法问题"""
        import re
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修复表格中的加号
            content = re.sub(r'(\|\s*[^|]*?)(\d+)\+([^|]*?\|)', r'\1\2及以上\3', content)
            
            # 修复表格中的星号
            content = re.sub(r'(\|\s*[^|]*?)(\d+)\*([^|]*?\|)', r'\1\2倍\3', content)
            
            # 修复表格中的尖括号 - 转换为HTML实体
            content = re.sub(r'(\|\s*[^|]*?)<([^|]*?\|)', r'\1&lt;\2', content)
            content = re.sub(r'(\|\s*[^|]*?)>([^|]*?\|)', r'\1&gt;\2', content)
            
            # 修复表格中的花括号
            content = re.sub(r'(\|\s*[^|]*?)\{([^|}]*?)\}([^|]*?\|)', r'\1\\{\2\\}\3', content)
            
            if content != original_content:
                if backup:
                    backup_path = str(file_path) + '.bak'
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return True
        
        except Exception as e:
            print(f"警告: 无法修复文件 {file_path} ({e})")
        
        return False


class MarkdownHeadingChecker:
    """检查和修复markdown标题结构问题"""
    
    def __init__(self):
        # 需要检查的模式
        self.long_content_threshold = 1000  # 长文档阈值
        self.min_h2_headings = 2  # 最少二级标题数量
    
    def scan_file(self, file_path):
        """扫描文件，查找标题结构问题"""
        import re
        issues = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 跳过 frontmatter
            lines = content.split('\n')
            content_start = 0
            if lines and lines[0].strip() == '---':
                # 寻找frontmatter结束
                for i, line in enumerate(lines[1:], 1):
                    if line.strip() == '---':
                        content_start = sum(len(l) + 1 for l in lines[:i+1])
                        break
                if content_start > 0:
                    content = content[content_start:]
                    lines = content.split('\n')
            
            # 检查是否有一级标题需要降级（排除代码块）
            content_lines = lines if content_start == 0 else content.split('\n')
            has_h1 = False
            in_code_block = False
            for line in content_lines:
                stripped = line.strip()
                # 检查代码块边界
                if stripped.startswith('```'):
                    in_code_block = not in_code_block
                    continue
                # 跳过代码块内的内容
                if in_code_block:
                    continue
                # 检查一级标题
                if re.match(r'^# [^#]', stripped):  # 匹配 "# " 开头但后面不是 #
                    has_h1 = True
                    break
            
            if has_h1:
                issues['需要降级一级标题'] = [(0, "检测到一级标题，需要降级以支持导航栏显示")]
            
            # 只对长文档进行其他检查
            if len(content) >= self.long_content_threshold:
                # 统计各级标题
                h1_count = 0
                h2_count = 0
                h3_count = 0
                escaped_bold_count = 0
                
                h1_lines = []
                potential_headings = []  # 可能应该是标题的粗体文本
                escaped_bold_lines = []
                
                # 统计标题时也要排除代码块
                in_code_block = False
                for line_num, line in enumerate(lines, 1):
                    stripped = line.strip()
                    
                    # 检查代码块边界
                    if stripped.startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    
                    # 跳过代码块内的内容
                    if in_code_block:
                        continue
                    
                    # 统计标题
                    if stripped.startswith('# '):
                        h1_count += 1
                        h1_lines.append((line_num + content_start // len('\n'), stripped))
                    elif stripped.startswith('## '):
                        h2_count += 1
                    elif stripped.startswith('### '):
                        h3_count += 1
                    
                    # 检查转义的粗体格式
                    if r'\*\*' in stripped:
                        escaped_bold_count += 1
                        escaped_bold_lines.append((line_num + content_start // len('\n'), stripped))
                    
                    # 检查可能应该是标题的粗体文本
                    if re.match(r'^\*\*[^*]+:\*\*', stripped) or re.match(r'^\*\*[^*]+\*\*$', stripped):
                        potential_headings.append((line_num + content_start // len('\n'), stripped))
                
                # 判断问题
                if h1_count > 3 and h2_count < self.min_h2_headings:
                    issues['缺少二级标题层级结构'] = [(0, f"文档有{h1_count}个一级标题，但只有{h2_count}个二级标题，建议增加层级结构")]
                
                if escaped_bold_count > 0:
                    issues['错误的粗体转义格式'] = escaped_bold_lines[:5]  # 最多显示5个
                
                if len(potential_headings) > 2:
                    issues['可能应该改为标题的粗体文本'] = potential_headings[:5]  # 最多显示5个
        
        except Exception as e:
            print(f"警告: 无法扫描文件 {file_path} ({e})")
        
        return issues
    
    def fix_file(self, file_path, backup=False):
        """修复文件中的标题结构问题"""
        import re
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修复内容，但排除代码块
            content = self._fix_content_excluding_code_blocks(content)
            
            # 降级一级标题（如果存在）
            content = self._downgrade_headings_if_needed(content)
            
            # 全局调整标题级别
            content, _ = self._adjust_heading_levels_globally(content)
            
            if content != original_content:
                if backup:
                    backup_path = str(file_path) + '.bak'
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return True
        
        except Exception as e:
            print(f"警告: 无法修复文件 {file_path} ({e})")
        
        return False
    
    def _fix_content_excluding_code_blocks(self, content):
        """修复内容，但排除代码块"""
        import re
        
        lines = content.split('\n')
        in_code_block = False
        
        for i, line in enumerate(lines):
            # 检查代码块边界
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            
            # 跳过代码块内的内容
            if in_code_block:
                continue
            
            # 修复转义的粗体格式
            line = re.sub(r'\\?\*\\?\*([^*]+)\\?\*\\?\*', r'**\1**', line)
            
            # 修复粗体后缺少空格的问题
            line = re.sub(r'\*\*([^*]+):\*\*([^\s])', r'**\1:** \2', line)
            
            # 将特定的粗体标记模式转换为标题
            # 匹配如 "**实现细节:**" 这样在单独行上的模式，转换为二级标题
            line = re.sub(r'^\*\*([^*]*(?:实现|特点|优势|机制|影响|解析|场景|支持|优化|问题|细节|方法|策略|原理)[^*]*)\*\*\s*$', r'## \1', line)
            
            # 匹配如 "**概念:**" 这样带冒号的模式
            line = re.sub(r'^\*\*([^*]+):\*\*\s*$', r'## \1', line)
            
            # 清理转换后可能产生的多余冒号
            line = re.sub(r'^## ([^:]+):\s*$', r'## \1', line)
            
            lines[i] = line
        
        return '\n'.join(lines)
    
    def _downgrade_headings_if_needed(self, content):
        """检查是否需要降级标题，如果有一级标题就将所有标题降一级"""
        import re
        
        # 跳过 frontmatter
        lines = content.split('\n')
        content_start = 0
        if lines and lines[0].strip() == '---':
            # 寻找frontmatter结束
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '---':
                    content_start = i + 1
                    break
        
        # 获取实际内容部分
        content_lines = lines[content_start:]
        
        # 检查是否有一级标题（# 开头但不是 ## 或更多#）
        has_h1 = False
        for line in content_lines:
            line = line.strip()
            if re.match(r'^# [^#]', line):  # 匹配 "# " 开头但后面不是 #
                has_h1 = True
                break
        
        if not has_h1:
            return content
        
        # 如果有一级标题，需要降级所有标题
        print("  检测到一级标题，将所有标题降级以支持导航栏显示")
        
        # 重新组合内容
        result_lines = lines[:content_start]  # 保留 frontmatter
        
        for line in content_lines:
            # 降级标题：# -> ##, ## -> ###, ### -> ####, 等等
            if re.match(r'^#{1,5} ', line):
                # 在开头添加一个 #
                result_lines.append('#' + line)
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines)

    def _adjust_heading_levels_globally(self, content: str) -> tuple[str, bool]:
        """
        全局调整标题级别：
        - 如果最小标题级别是###或更高，全局删掉一个#
        - 如果最小标题级别是#，全局添加一个#
        """
        lines = content.split('\n')
        
        # 跳过 frontmatter
        content_start = 0
        if lines and lines[0].strip() == '---':
            # 寻找frontmatter结束
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '---':
                    content_start = i + 1
                    break
        
        # 获取实际内容部分
        content_lines = lines[content_start:]
        heading_lines = []
        
        # 跟踪代码块状态
        in_code_block = False
        
        # 找出所有标题行和它们的级别
        for i, line in enumerate(content_lines):
            stripped = line.strip()
            
            # 检查代码块边界
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                continue
            
            # 跳过代码块内的内容
            if in_code_block:
                continue
            
            # 检查是否是真正的标题（行首是#，且不在代码块中）
            if stripped.startswith('#'):
                # 计算标题级别
                level = 0
                for char in stripped:
                    if char == '#':
                        level += 1
                    else:
                        break
                
                # 确保#后面有空格，这是标准的markdown标题格式
                # 支持1-6级标题，但我们会将超过4级的标题调整到合理范围内
                if level > 0 and level <= 6 and len(stripped) > level and stripped[level] == ' ':
                    heading_lines.append((i, level, stripped))
        
        if not heading_lines:
            return content, False
            
        # 找到最小标题级别
        min_level = min(level for _, level, _ in heading_lines)
        
        modified = False
        
        # 决定调整策略
        if min_level >= 3:
            # 最小级别是###或更高，全局删掉一个#
            for i, level, heading_text in heading_lines:
                if level > 1:  # 确保不会变成0级标题
                    new_level = max(1, level - 1)  # 最少是1级标题
                    new_heading = '#' * new_level + heading_text[level:]
                    # 只替换行首的标题，使用更精确的匹配
                    if content_lines[i].strip().startswith(heading_text):
                        content_lines[i] = content_lines[i].replace(heading_text, new_heading, 1)  # 只替换第一个匹配
                        modified = True
        elif min_level == 1:
            # 最小级别是#，全局添加一个#
            for i, level, heading_text in heading_lines:
                new_level = min(4, level + 1)  # 最多是4级标题
                new_heading = '#' * new_level + heading_text[level:]
                # 只替换行首的标题，使用更精确的匹配
                if content_lines[i].strip().startswith(heading_text):
                    content_lines[i] = content_lines[i].replace(heading_text, new_heading, 1)  # 只替换第一个匹配
                    modified = True
        
        # 额外处理：将所有超过4级的标题调整为4级
        for i, level, heading_text in heading_lines:
            if level > 4:
                new_heading = '#### ' + heading_text[level:].lstrip()
                # 只替换行首的标题，使用更精确的匹配
                if content_lines[i].strip().startswith(heading_text):
                    content_lines[i] = content_lines[i].replace(heading_text, new_heading, 1)  # 只替换第一个匹配
                    modified = True
        
        # 重新组合完整内容
        result_lines = lines[:content_start] + content_lines
        return '\n'.join(result_lines), modified


class PreCommitChecker:
    def __init__(self):
        self.mdx_fixer = MDXTableFixer()
        self.punct_fixer = MarkdownPunctuationFixer()
        self.mdx_syntax_checker = MDXSyntaxChecker()
        self.heading_checker = MarkdownHeadingChecker()
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
                    # 重新扫描确认修复效果
                    recheck_issues = self.punct_fixer.scan_file(file_path)
                    if not recheck_issues:
                        has_issues = False  # 确认已修复
                    else:
                        print(f"  ⚠️ 修复后仍有问题，可能在代码块中")
                        has_issues = True
        
        # MDX语法检查
        mdx_syntax_issues = self.mdx_syntax_checker.scan_file(file_path)
        if mdx_syntax_issues:
            has_issues = True
            for issue_type, issue_list in mdx_syntax_issues.items():
                print(f"  ❌ {issue_type}: {len(issue_list)} 个问题")
                if not fix_mode:
                    for line_num, line_content in issue_list[:3]:
                        print(f"    第{line_num}行: {line_content[:60]}{'...' if len(line_content) > 60 else ''}")
            
            if fix_mode:
                if self.mdx_syntax_checker.fix_file(file_path):
                    print(f"  ✓ MDX语法问题已修复")
                    has_issues = False  # 已修复
        
        # 标题结构检查
        heading_issues = self.heading_checker.scan_file(file_path)
        if heading_issues:
            has_issues = True
            for issue_type, issue_list in heading_issues.items():
                print(f"  ❌ {issue_type}: {len(issue_list)} 个问题")
                if not fix_mode:
                    for line_num, line_content in issue_list[:3]:
                        print(f"    第{line_num}行: {line_content[:60]}{'...' if len(line_content) > 60 else ''}")
            
            if fix_mode:
                if self.heading_checker.fix_file(file_path):
                    print(f"  ✓ 标题结构问题已修复")
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