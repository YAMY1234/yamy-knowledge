#!/usr/bin/env python3
"""
Pre-commit 检查脚本

集成多种检查功能:
1. MDX表格数字格式检查
2. 中文标点检查
3. MDX语法特殊字符检查
4. 数学公式格式检查（\$转$$）
5. Markdown标题结构检查
6. 支持Git hook使用

使用方法:
python3 scripts/pre-commit-checks.py [--fix] [--staged-only]

选项:
--fix: 自动修复问题
--staged-only: 只检查已暂存的文件

功能配置:
可以通过修改脚本开头的 FEATURE_CONFIG 字典来启用/禁用各项功能：
- mdx_table_check: MDX表格检查（默认开启）
- punctuation_check: 中文标点检查（默认开启）
- mdx_syntax_check: MDX语法特殊字符检查（默认开启）
- math_formula_check: 数学公式格式检查，将\$...\$转换为$$...$$格式（默认开启）
- escaped_bold_fix: 转义粗体格式修复，将\*\*转换为**（默认开启）
- heading_structure_check: 标题结构检查（默认关闭）
- heading_level_adjustment: 标题级别调整/批量增减#（默认关闭）
- bold_to_heading_conversion: 粗体转标题功能（默认关闭）
- heading_downgrade: 一级标题降级功能（默认关闭）
- details_heading_conversion: details块中标题转粗体格式（默认开启）
- bold_spacing_fix: 修复粗体标记边界空格问题（默认开启）
- bold_surrounding_spacing: 确保粗体文本两边有适当空格（默认开启）

标题结构检查功能（已默认关闭）:
- 检测长文档缺少二级标题层级结构的问题
- 检测和修复错误的粗体转义格式 (\*\*文本\*\*)
- 自动将常见的粗体文本模式转换为合适的标题格式
- 全局调整标题级别（批量增减#号）

数学公式格式检查功能（已默认开启）:
- 检测并修复\$...\$格式的数学公式，转换为$$...$$格式
- 避免修改代码块中的内容
- 保护行内代码块不被修改

转义粗体格式修复功能（已默认开启）:
- 检测并修复转义的粗体格式\*\*...\*\*，转换为**...**格式
- 避免修改代码块中的内容
- 在其他粗体格式检查之前执行
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Set

# ======================== 功能开关配置 ========================
# 在这里配置各种检查功能的开关
FEATURE_CONFIG = {
    # MDX 和标点检查（推荐保持开启）
    'mdx_table_check': True,           # MDX表格检查
    'punctuation_check': True,         # 中文标点检查
    'mdx_syntax_check': True,          # MDX语法特殊字符检查
    'math_formula_check': True,        # 数学公式格式检查（\$转$$）
    'escaped_bold_fix': True,          # 转义粗体格式修复（\*\*转**）
    
    # 标题相关检查（可以关闭）
    'heading_structure_check': False,   # 标题结构检查（缺少二级标题等）
    'heading_level_adjustment': False,  # 标题级别调整（批量增减#）
    'bold_to_heading_conversion': False, # 粗体转标题功能
    'heading_downgrade': False,         # 一级标题降级功能
    'details_heading_conversion': True,  # details块中标题转粗体格式（默认开启）
    'bold_spacing_fix': True,           # 修复粗体标记边界空格问题（默认开启）
    'bold_surrounding_spacing': True,   # 确保粗体文本两边有适当空格（默认开启）
}
# ============================================================

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
        
        # 如果标题结构检查被禁用，直接返回空结果
        if not FEATURE_CONFIG.get('heading_structure_check', False):
            return issues
        
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
            if FEATURE_CONFIG.get('heading_downgrade', False):
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
                    
                    # 检查可能应该是标题的粗体文本（仅当功能启用时）
                    if FEATURE_CONFIG.get('bold_to_heading_conversion', False):
                        # 更严格地检查可能应该是标题的粗体文本
                        # 只有当粗体文本独占一行，且不是以冒号结尾的概念定义时，才认为可能是标题
                        if (re.match(r'^\*\*[^*]+\*\*$', stripped) and  # 独行的粗体文本
                            not re.match(r'^\*\*[^*]+:\*\*', stripped) and  # 排除概念定义格式（以冒号结尾）
                            not line.strip().endswith(':')):  # 排除其他以冒号结尾的情况
                            potential_headings.append((line_num + content_start // len('\n'), stripped))
                
                # 判断问题
                if h1_count > 3 and h2_count < self.min_h2_headings:
                    issues['缺少二级标题层级结构'] = [(0, f"文档有{h1_count}个一级标题，但只有{h2_count}个二级标题，建议增加层级结构")]
                
                if escaped_bold_count > 0:
                    issues['错误的粗体转义格式'] = escaped_bold_lines[:5]  # 最多显示5个
                
                # 只有在功能启用且有足够多的符合条件的粗体文本时才报告问题
                if FEATURE_CONFIG.get('bold_to_heading_conversion', False) and len(potential_headings) > 3:
                    issues['可能应该改为标题的粗体文本'] = potential_headings[:5]  # 最多显示5个
        
        except Exception as e:
            print(f"警告: 无法扫描文件 {file_path} ({e})")
        
        return issues
    
    def fix_file(self, file_path, backup=False):
        """修复文件中的标题结构问题"""
        import re
        
        # 如果标题结构检查被禁用，直接返回False
        if not FEATURE_CONFIG.get('heading_structure_check', False):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修复内容，但排除代码块（根据配置决定）
            if FEATURE_CONFIG.get('bold_to_heading_conversion', False):
                content = self._fix_content_excluding_code_blocks(content)
            
            # 降级一级标题（根据配置决定）
            if FEATURE_CONFIG.get('heading_downgrade', False):
                content = self._downgrade_headings_if_needed(content)
            
            # 全局调整标题级别（根据配置决定）
            if FEATURE_CONFIG.get('heading_level_adjustment', False):
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
            
            # 将特定的粗体标记模式转换为标题（根据配置决定）
            if FEATURE_CONFIG.get('bold_to_heading_conversion', False):
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


class DetailsHeadingConverter:
    """转换details块中的标题格式为粗体格式"""
    
    def __init__(self):
        pass
    
    def scan_file(self, file_path):
        """扫描文件中details块内的标题问题"""
        import re
        issues = {}
        
        # 如果details标题转换功能被禁用，直接返回空结果
        if not FEATURE_CONFIG.get('details_heading_conversion', True):
            return issues
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 找到所有details块
            details_blocks = self._find_details_blocks(content)
            
            for block_start, block_end, block_content in details_blocks:
                # 在details块中查找标题
                headings_found = self._find_headings_in_content(block_content)
                
                if headings_found:
                    if 'details块中的标题格式' not in issues:
                        issues['details块中的标题格式'] = []
                    
                    for line_offset, heading_text in headings_found:
                        # 计算在整个文件中的行号
                        lines_before_block = content[:block_start].count('\n')
                        actual_line_num = lines_before_block + line_offset + 1
                        issues['details块中的标题格式'].append((actual_line_num, heading_text.strip()))
        
        except Exception as e:
            print(f"警告: 无法扫描文件 {file_path} ({e})")
        
        return issues
    
    def fix_file(self, file_path, backup=False):
        """修复文件中details块内的标题格式"""
        import re
        
        # 如果details标题转换功能被禁用，直接返回False
        if not FEATURE_CONFIG.get('details_heading_conversion', True):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 转换details块中的标题
            content = self._convert_headings_in_details(content)
            
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
    
    def _find_details_blocks(self, content):
        """找到所有details块的位置和内容"""
        import re
        
        details_blocks = []
        
        # 使用正则表达式找到所有details块
        pattern = r'<details>(.*?)</details>'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            start_pos = match.start()
            end_pos = match.end()
            block_content = match.group(1)
            details_blocks.append((start_pos, end_pos, block_content))
        
        return details_blocks
    
    def _find_headings_in_content(self, content):
        """在给定内容中查找标题，排除代码块"""
        import re
        
        lines = content.split('\n')
        headings = []
        in_code_block = False
        
        for line_num, line in enumerate(lines):
            stripped = line.strip()
            
            # 检查代码块边界
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                continue
            
            # 跳过代码块内的内容
            if in_code_block:
                continue
            
            # 检查是否是标题（## 或 ###）
            if re.match(r'^#{2,3}\s+', stripped):
                headings.append((line_num, line))
        
        return headings
    
    def _convert_headings_in_details(self, content):
        """转换details块中的标题格式"""
        import re
        
        def convert_details_block(match):
            details_content = match.group(1)
            
            # 在details内容中转换标题
            converted_content = self._convert_headings_in_block_content(details_content)
            
            return f'<details>{converted_content}</details>'
        
        # 处理所有details块
        pattern = r'<details>(.*?)</details>'
        return re.sub(pattern, convert_details_block, content, flags=re.DOTALL)
    
    def _convert_headings_in_block_content(self, content):
        """转换块内容中的标题，排除代码块"""
        import re
        
        lines = content.split('\n')
        result_lines = []
        in_code_block = False
        
        for line in lines:
            stripped = line.strip()
            
            # 检查代码块边界
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                result_lines.append(line)
                continue
            
            # 跳过代码块内的内容
            if in_code_block:
                result_lines.append(line)
                continue
            
            # 转换标题格式
            # 将 ### 标题转换为 **标题**
            line = re.sub(r'^(\s*)###\s+(.+)$', r'\1**\2**', line)
            # 将 ## 标题转换为 **标题**
            line = re.sub(r'^(\s*)##\s+(.+)$', r'\1**\2**', line)
            
            result_lines.append(line)
        
        return '\n'.join(result_lines)


class BoldSpacingFixer:
    """修复粗体标记边界处的空格问题"""
    
    def __init__(self):
        pass
    
    def scan_file(self, file_path):
        """扫描文件中粗体边界空格问题"""
        import re
        issues = {}
        
        # 如果粗体空格修复功能被禁用，直接返回空结果
        if not FEATURE_CONFIG.get('bold_spacing_fix', True):
            return issues
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 排除代码块后查找问题
            problematic_patterns = self._find_bold_spacing_issues(content)
            
            if problematic_patterns:
                issues['粗体标记边界空格问题'] = problematic_patterns
        
        except Exception as e:
            print(f"警告: 无法扫描文件 {file_path} ({e})")
        
        return issues
    
    def fix_file(self, file_path, backup=False):
        """修复文件中的粗体边界空格问题"""
        import re
        
        # 如果粗体空格修复功能被禁用，直接返回False
        if not FEATURE_CONFIG.get('bold_spacing_fix', True):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修复粗体边界空格问题
            content = self._fix_bold_spacing_in_content(content)
            
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
    
    def _find_bold_spacing_issues(self, content):
        """查找粗体边界空格问题，排除代码块"""
        import re
        
        lines = content.split('\n')
        issues = []
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
            
            # 使用新的算法检查粗体块配对
            line_issues = self._check_bold_colon_spacing_in_line(line)
            if line_issues:
                issues.append((line_num, line.strip()[:80] + ('...' if len(line.strip()) > 80 else '')))
        
        return issues
    
    def _check_bold_colon_spacing_in_line(self, text):
        """检查单行中是否有需要修复的粗体冒号空格问题"""
        import re
        
        issues = []
        i = 0
        
        while i < len(text):
            # 查找下一个 **
            if i < len(text) - 1 and text[i:i+2] == '**':
                start = i
                i += 2  # 跳过开始的 **
                
                # 查找对应的结束 **
                bold_content = []
                while i < len(text) - 1:
                    if text[i:i+2] == '**':
                        # 找到结束的 **
                        bold_text = ''.join(bold_content)
                        
                        # 检查这个粗体块是否有问题
                        # 1. 以冒号+空格结尾
                        # 2. 开头有空格
                        # 3. 结尾有空格（非冒号）
                        if (re.match(r'^.+:\s+$', bold_text) or  # 冒号+空格结尾
                            re.match(r'^\s+.+$', bold_text) or   # 开头有空格
                            re.match(r'^.+\s+$', bold_text)):    # 结尾有空格
                            issues.append(f"**{bold_text}**")
                        
                        i += 2  # 跳过结束的 **
                        break
                    else:
                        bold_content.append(text[i])
                        i += 1
                else:
                    # 没有找到匹配的结束 **
                    i = start + 1
            else:
                i += 1
        
        return issues
    
    def _fix_bold_spacing_in_content(self, content):
        """修复内容中的粗体边界空格问题，排除代码块"""
        import re
        
        lines = content.split('\n')
        result_lines = []
        in_code_block = False
        
        for line in lines:
            stripped = line.strip()
            
            # 检查代码块边界
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                result_lines.append(line)
                continue
            
            # 跳过代码块内的内容
            if in_code_block:
                result_lines.append(line)
                continue
            
            # 使用新的算法修复粗体边界空格问题
            fixed_line = self._fix_bold_colon_spacing_in_line(line)
            
            # 清理可能产生的多余空格
            fixed_line = re.sub(r'  +', ' ', fixed_line)  # 多个空格合并为一个
            
            result_lines.append(fixed_line)
        
        return '\n'.join(result_lines)
    
    def _fix_bold_colon_spacing_in_line(self, text):
        """修复单行中的粗体冒号空格问题"""
        import re
        
        result = []
        i = 0
        
        while i < len(text):
            # 查找下一个 **
            if i < len(text) - 1 and text[i:i+2] == '**':
                # 找到粗体块的开始
                start = i
                i += 2  # 跳过开始的 **
                
                # 查找对应的结束 **
                bold_content = []
                while i < len(text) - 1:
                    if text[i:i+2] == '**':
                        # 找到结束的 **
                        bold_text = ''.join(bold_content)
                        
                        # 检查这个粗体块是否需要修复
                        if re.match(r'^.+:\s+$', bold_text):
                            # 冒号+空格结尾：移除冒号后的空格，然后在粗体外添加空格
                            fixed_content = re.sub(r':\s+$', ':', bold_text)
                            result.append('**' + fixed_content + '** ')
                        elif re.match(r'^\s+(.+)$', bold_text):
                            # 开头有空格：移除开头空格，在粗体外添加空格
                            fixed_content = re.sub(r'^\s+', '', bold_text)
                            result.append(' **' + fixed_content + '**')
                        elif re.match(r'^(.+)\s+$', bold_text) and not re.match(r'^.+:\s+$', bold_text):
                            # 结尾有空格（非冒号）：移除结尾空格，在粗体外添加空格
                            fixed_content = re.sub(r'\s+$', '', bold_text)
                            result.append('**' + fixed_content + '** ')
                        else:
                            # 不需要修复，保持原样
                            result.append('**' + bold_text + '**')
                        
                        i += 2  # 跳过结束的 **
                        break
                    else:
                        bold_content.append(text[i])
                        i += 1
                else:
                    # 没有找到匹配的结束 **，保持原样
                    result.append(text[start:i])
            else:
                result.append(text[i])
                i += 1
        
        return ''.join(result)


class BoldSurroundingSpacingFixer:
    """确保粗体文本两边有适当的空格"""
    
    def __init__(self):
        # 标点符号集合（中文和英文）
        self.punctuation = set('.,;:!?，。；：！？、（）()[]{}「」『』""''""''…—–-')
    
    def scan_file(self, file_path):
        """扫描文件中粗体文本周围空格问题"""
        import re
        issues = {}
        
        # 如果粗体周围空格功能被禁用，直接返回空结果
        if not FEATURE_CONFIG.get('bold_surrounding_spacing', True):
            return issues
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 排除代码块后查找问题
            problematic_patterns = self._find_bold_surrounding_issues(content)
            
            if problematic_patterns:
                issues['粗体文本周围空格问题'] = problematic_patterns
        
        except Exception as e:
            print(f"警告: 无法扫描文件 {file_path} ({e})")
        
        return issues
    
    def fix_file(self, file_path, backup=False):
        """修复文件中的粗体文本周围空格问题"""
        import re
        
        # 如果粗体周围空格功能被禁用，直接返回False
        if not FEATURE_CONFIG.get('bold_surrounding_spacing', True):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修复粗体文本周围空格问题
            content = self._fix_bold_surrounding_spacing(content)
            
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
    
    def _find_bold_surrounding_issues(self, content):
        """查找粗体文本周围空格问题，排除代码块"""
        import re
        
        lines = content.split('\n')
        issues = []
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
            
            # 检查这一行是否有粗体周围空格问题
            if self._has_bold_surrounding_issues(line):
                issues.append((line_num, line.strip()[:80] + ('...' if len(line.strip()) > 80 else '')))
        
        return issues
    
    def _has_bold_surrounding_issues(self, text):
        """检查单行中是否有粗体周围空格问题"""
        import re
        
        # 查找所有完整的粗体块
        bold_pattern = r'\*\*[^*]+\*\*'
        matches = list(re.finditer(bold_pattern, text))
        
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            
            # 分别检查前面和后面是否需要空格
            need_space_before = False
            need_space_after = False
            
            # 检查前面是否需要空格
            if start_pos > 0:
                prev_char = text[start_pos - 1]
                # 如果前面不是空格、标点符号，则需要空格
                if prev_char not in self.punctuation and prev_char != ' ':
                    need_space_before = True
            
            # 检查后面是否需要空格
            if end_pos < len(text):
                next_char = text[end_pos]
                # 如果后面不是空格、标点符号，则需要空格
                if next_char not in self.punctuation and next_char != ' ':
                    need_space_after = True
            
            # 只要有一边需要空格就返回True
            if need_space_before or need_space_after:
                return True
        
        return False
    
    def _fix_bold_surrounding_spacing(self, content):
        """修复内容中的粗体文本周围空格问题，排除代码块"""
        import re
        
        lines = content.split('\n')
        result_lines = []
        in_code_block = False
        
        for line in lines:
            stripped = line.strip()
            
            # 检查代码块边界
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                result_lines.append(line)
                continue
            
            # 跳过代码块内的内容
            if in_code_block:
                result_lines.append(line)
                continue
            
            # 修复这一行的粗体周围空格问题
            fixed_line = self._fix_bold_surrounding_spacing_in_line(line)
            result_lines.append(fixed_line)
        
        return '\n'.join(result_lines)
    
    def _fix_bold_surrounding_spacing_in_line(self, text):
        """修复单行中的粗体周围空格问题"""
        import re
        
        # 查找所有完整的粗体块，从后往前处理以避免位置偏移
        bold_pattern = r'\*\*[^*]+\*\*'
        matches = list(re.finditer(bold_pattern, text))
        
        # 从后往前处理，避免位置偏移
        for match in reversed(matches):
            start_pos = match.start()
            end_pos = match.end()
            bold_text = match.group()
            
            # 检查并修复前面的空格
            need_space_before = False
            if start_pos > 0:
                prev_char = text[start_pos - 1]
                if prev_char not in self.punctuation and prev_char != ' ':
                    need_space_before = True
            
            # 检查并修复后面的空格
            need_space_after = False
            if end_pos < len(text):
                next_char = text[end_pos]
                if next_char not in self.punctuation and next_char != ' ':
                    need_space_after = True
            
            # 构建修复后的文本
            if need_space_before or need_space_after:
                new_bold_text = bold_text
                if need_space_before:
                    new_bold_text = ' ' + new_bold_text
                if need_space_after:
                    new_bold_text = new_bold_text + ' '
                
                # 替换原文本
                text = text[:start_pos] + new_bold_text + text[end_pos:]
        
        return text


class MathFormulaChecker:
    """数学公式格式检查器，检查并修复\$...\$格式为$$...$$格式"""
    
    def __init__(self):
        # 匹配单个$包围的数学公式的正则表达式
        # 避免匹配代码块中的内容和已经是$$格式的公式
        import re
        # 匹配 \$...\$ 格式的公式（转义的$符号），但不匹配已经是$$...$$格式的
        # 使用负向前瞻和负向后瞻确保前后没有$符号
        self.escaped_formula_pattern = re.compile(r'(?<!\$)\\?\$([^$]+?)\\?\$(?!\$)')
        
    def scan_file(self, file_path):
        """扫描文件中的数学公式格式问题"""
        import re
        issues = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            in_code_block = False
            code_block_pattern = re.compile(r'^```')
            
            for line_num, line in enumerate(lines, 1):
                # 检测代码块边界
                if code_block_pattern.match(line):
                    in_code_block = not in_code_block
                    continue
                
                # 跳过代码块内的内容
                if in_code_block:
                    continue
                
                # 跳过行内代码
                line_without_inline_code = re.sub(r'`[^`]*`', '', line)
                
                # 查找 \$...\$ 格式的公式
                matches = self.escaped_formula_pattern.findall(line_without_inline_code)
                if matches:
                    issue_type = "数学公式格式需要从\\$...\\$转换为$$...$$"
                    if issue_type not in issues:
                        issues[issue_type] = []
                    for match in matches:
                        issues[issue_type].append((line_num, line.strip()))
        
        except Exception as e:
            print(f"警告: 无法扫描文件 {file_path} ({e})")
        
        return issues
    
    def fix_file(self, file_path, backup=False):
        """修复文件中的数学公式格式问题"""
        import re
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 分割内容为行，逐行处理以避免修改代码块
            lines = content.split('\n')
            in_code_block = False
            code_block_pattern = re.compile(r'^```')
            
            fixed_lines = []
            for line in lines:
                # 检测代码块边界
                if code_block_pattern.match(line):
                    in_code_block = not in_code_block
                    fixed_lines.append(line)
                    continue
                
                # 跳过代码块内的内容
                if in_code_block:
                    fixed_lines.append(line)
                    continue
                
                # 保护行内代码块
                inline_code_parts = []
                temp_line = line
                
                # 提取行内代码块
                inline_code_pattern = re.compile(r'`[^`]*`')
                for match in inline_code_pattern.finditer(line):
                    placeholder = f"__INLINE_CODE_{len(inline_code_parts)}__"
                    inline_code_parts.append(match.group())
                    temp_line = temp_line.replace(match.group(), placeholder, 1)
                
                # 在非代码块内容中进行替换
                # 将 \$...\$ 替换为 $$...$$，但不匹配已经是$$...$$格式的
                temp_line = re.sub(r'(?<!\$)\\?\$([^$]+?)\\?\$(?!\$)', r'$$\1$$', temp_line)
                
                # 还原行内代码块
                for i, code_block in enumerate(inline_code_parts):
                    temp_line = temp_line.replace(f"__INLINE_CODE_{i}__", code_block)
                
                fixed_lines.append(temp_line)
            
            fixed_content = '\n'.join(fixed_lines)
            
            if fixed_content != original_content:
                if backup:
                    backup_path = str(file_path) + '.bak'
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                return True
        
        except Exception as e:
            print(f"警告: 无法修复文件 {file_path} ({e})")
        
        return False


class EscapedBoldFixer:
    """转义粗体格式修复器，将\*\*...\*\*格式转换为**...**格式"""
    
    def __init__(self):
        import re
        # 匹配转义的粗体格式 \*\*...\*\*
        self.escaped_bold_pattern = re.compile(r'\\?\*\\?\*([^*]+?)\\?\*\\?\*')
        
    def scan_file(self, file_path):
        """扫描文件中的转义粗体格式问题"""
        import re
        issues = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            in_code_block = False
            code_block_pattern = re.compile(r'^```')
            
            for line_num, line in enumerate(lines, 1):
                # 检测代码块边界
                if code_block_pattern.match(line):
                    in_code_block = not in_code_block
                    continue
                
                # 跳过代码块内的内容
                if in_code_block:
                    continue
                
                # 跳过行内代码
                line_without_inline_code = re.sub(r'`[^`]*`', '', line)
                
                # 查找转义的粗体格式
                matches = self.escaped_bold_pattern.findall(line_without_inline_code)
                if matches:
                    issue_type = "转义粗体格式需要从\\*\\*...\\*\\*转换为**...**"
                    if issue_type not in issues:
                        issues[issue_type] = []
                    for match in matches:
                        issues[issue_type].append((line_num, line.strip()))
        
        except Exception as e:
            print(f"警告: 无法扫描文件 {file_path} ({e})")
        
        return issues
    
    def fix_file(self, file_path, backup=False):
        """修复文件中的转义粗体格式问题"""
        import re
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 分割内容为行，逐行处理以避免修改代码块
            lines = content.split('\n')
            in_code_block = False
            code_block_pattern = re.compile(r'^```')
            
            fixed_lines = []
            for line in lines:
                # 检测代码块边界
                if code_block_pattern.match(line):
                    in_code_block = not in_code_block
                    fixed_lines.append(line)
                    continue
                
                # 跳过代码块内的内容
                if in_code_block:
                    fixed_lines.append(line)
                    continue
                
                # 保护行内代码块
                inline_code_parts = []
                temp_line = line
                
                # 提取行内代码块
                inline_code_pattern = re.compile(r'`[^`]*`')
                for match in inline_code_pattern.finditer(line):
                    placeholder = f"__INLINE_CODE_{len(inline_code_parts)}__"
                    inline_code_parts.append(match.group())
                    temp_line = temp_line.replace(match.group(), placeholder, 1)
                
                # 在非代码块内容中进行替换
                # 将 \*\*...\*\* 替换为 **...**
                temp_line = re.sub(r'\\?\*\\?\*([^*]+?)\\?\*\\?\*', r'**\1**', temp_line)
                
                # 还原行内代码块
                for i, code_block in enumerate(inline_code_parts):
                    temp_line = temp_line.replace(f"__INLINE_CODE_{i}__", code_block)
                
                fixed_lines.append(temp_line)
            
            fixed_content = '\n'.join(fixed_lines)
            
            if fixed_content != original_content:
                if backup:
                    backup_path = str(file_path) + '.bak'
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                return True
        
        except Exception as e:
            print(f"警告: 无法修复文件 {file_path} ({e})")
        
        return False


class PreCommitChecker:
    def __init__(self):
        self.mdx_fixer = MDXTableFixer()
        self.punct_fixer = MarkdownPunctuationFixer()
        self.mdx_syntax_checker = MDXSyntaxChecker()
        self.heading_checker = MarkdownHeadingChecker()
        self.details_converter = DetailsHeadingConverter()
        self.bold_spacing_fixer = BoldSpacingFixer()
        self.bold_surrounding_fixer = BoldSurroundingSpacingFixer()
        self.math_formula_checker = MathFormulaChecker()
        self.escaped_bold_fixer = EscapedBoldFixer()
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
        
        # MDX表格检查（根据配置决定）
        if FEATURE_CONFIG.get('mdx_table_check', True):
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
        
        # 中文标点检查（根据配置决定）
        if FEATURE_CONFIG.get('punctuation_check', True):
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
        
        # MDX语法检查（根据配置决定）
        if FEATURE_CONFIG.get('mdx_syntax_check', True):
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
        
        # 标题结构检查（根据配置决定）
        if FEATURE_CONFIG.get('heading_structure_check', False):
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
        
        # Details块标题转换检查（根据配置决定）
        if FEATURE_CONFIG.get('details_heading_conversion', True):
            details_issues = self.details_converter.scan_file(file_path)
            if details_issues:
                has_issues = True
                for issue_type, issue_list in details_issues.items():
                    print(f"  ❌ {issue_type}: {len(issue_list)} 个问题")
                    if not fix_mode:
                        for line_num, line_content in issue_list[:3]:
                            print(f"    第{line_num}行: {line_content[:60]}{'...' if len(line_content) > 60 else ''}")
                
                if fix_mode:
                    if self.details_converter.fix_file(file_path):
                        print(f"  ✓ Details块标题格式已修复")
                        has_issues = False  # 已修复
        
        # 粗体边界空格检查（根据配置决定）
        if FEATURE_CONFIG.get('bold_spacing_fix', True):
            bold_spacing_issues = self.bold_spacing_fixer.scan_file(file_path)
            if bold_spacing_issues:
                has_issues = True
                for issue_type, issue_list in bold_spacing_issues.items():
                    print(f"  ❌ {issue_type}: {len(issue_list)} 个问题")
                    if not fix_mode:
                        for line_num, line_content in issue_list[:3]:
                            print(f"    第{line_num}行: {line_content[:60]}{'...' if len(line_content) > 60 else ''}")
                
                if fix_mode:
                    if self.bold_spacing_fixer.fix_file(file_path):
                        print(f"  ✓ 粗体边界空格问题已修复")
                        has_issues = False  # 已修复
        
        # 粗体文本周围空格检查（根据配置决定）
        if FEATURE_CONFIG.get('bold_surrounding_spacing', True):
            bold_surrounding_issues = self.bold_surrounding_fixer.scan_file(file_path)
            if bold_surrounding_issues:
                has_issues = True
                for issue_type, issue_list in bold_surrounding_issues.items():
                    print(f"  ❌ {issue_type}: {len(issue_list)} 个问题")
                    if not fix_mode:
                        for line_num, line_content in issue_list[:3]:
                            print(f"    第{line_num}行: {line_content[:60]}{'...' if len(line_content) > 60 else ''}")
                
                if fix_mode:
                    if self.bold_surrounding_fixer.fix_file(file_path):
                        print(f"  ✓ 粗体文本周围空格问题已修复")
                        has_issues = False  # 已修复
        
        # 数学公式格式检查（根据配置决定）
        if FEATURE_CONFIG.get('math_formula_check', True):
            math_formula_issues = self.math_formula_checker.scan_file(file_path)
            if math_formula_issues:
                has_issues = True
                for issue_type, issue_list in math_formula_issues.items():
                    print(f"  ❌ {issue_type}: {len(issue_list)} 个问题")
                    if not fix_mode:
                        for line_num, line_content in issue_list[:3]:
                            print(f"    第{line_num}行: {line_content[:60]}{'...' if len(line_content) > 60 else ''}")
                
                if fix_mode:
                    if self.math_formula_checker.fix_file(file_path):
                        print(f"  ✓ 数学公式格式已修复")
                        has_issues = False  # 已修复
        
        # 转义粗体格式检查（根据配置决定）- 必须在其他粗体检查之前执行
        if FEATURE_CONFIG.get('escaped_bold_fix', True):
            escaped_bold_issues = self.escaped_bold_fixer.scan_file(file_path)
            if escaped_bold_issues:
                has_issues = True
                for issue_type, issue_list in escaped_bold_issues.items():
                    print(f"  ❌ {issue_type}: {len(issue_list)} 个问题")
                    if not fix_mode:
                        for line_num, line_content in issue_list[:3]:
                            print(f"    第{line_num}行: {line_content[:60]}{'...' if len(line_content) > 60 else ''}")
                
                if fix_mode:
                    if self.escaped_bold_fixer.fix_file(file_path):
                        print(f"  ✓ 转义粗体格式已修复")
                        has_issues = False  # 已修复
        
        # 粗体边界空格检查（根据配置决定）
        if FEATURE_CONFIG.get('bold_spacing_fix', True):
            bold_spacing_issues = self.bold_spacing_fixer.scan_file(file_path)
            if bold_spacing_issues:
                has_issues = True
                for issue_type, issue_list in bold_spacing_issues.items():
                    print(f"  ❌ {issue_type}: {len(issue_list)} 个问题")
                    if not fix_mode:
                        for line_num, line_content in issue_list[:3]:
                            print(f"    第{line_num}行: {line_content[:60]}{'...' if len(line_content) > 60 else ''}")
                
                if fix_mode:
                    if self.bold_spacing_fixer.fix_file(file_path):
                        print(f"  ✓ 粗体边界空格问题已修复")
                        has_issues = False  # 已修复
        
        # 粗体文本周围空格检查（根据配置决定）
        if FEATURE_CONFIG.get('bold_surrounding_spacing', True):
            bold_surrounding_issues = self.bold_surrounding_fixer.scan_file(file_path)
            if bold_surrounding_issues:
                has_issues = True
                for issue_type, issue_list in bold_surrounding_issues.items():
                    print(f"  ❌ {issue_type}: {len(issue_list)} 个问题")
                    if not fix_mode:
                        for line_num, line_content in issue_list[:3]:
                            print(f"    第{line_num}行: {line_content[:60]}{'...' if len(line_content) > 60 else ''}")
                
                if fix_mode:
                    if self.bold_surrounding_fixer.fix_file(file_path):
                        print(f"  ✓ 粗体文本周围空格问题已修复")
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