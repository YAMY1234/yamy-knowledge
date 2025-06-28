#!/usr/bin/env python3
"""
自动创建LLM知识体系目录结构的脚本
"""

import os
import json
from pathlib import Path

def create_directory_structure():
    """创建完整的LLM知识体系目录结构"""
    
    # 基础路径
    base_path = Path("docs")
    
    # 目录结构定义
    directories_config = {
        "llm-security": {
            "label": "LLM安全与风险",
            "position": 4,
            "description": "探索LLM的安全风险、攻击防护与安全对齐。",
            "files": [
                {"name": "prompt-injection.md", "title": "提示注入攻击", "id": "prompt-injection"},
                {"name": "data-privacy.md", "title": "数据隐私保护", "id": "data-privacy"},
                {"name": "model-security.md", "title": "模型安全", "id": "model-security"},
                {"name": "adversarial-attacks.md", "title": "对抗性攻击", "id": "adversarial-attacks"},
                {"name": "safety-alignment.md", "title": "安全对齐", "id": "safety-alignment"}
            ]
        },
        "llm-evaluation": {
            "label": "LLM评估与测试",
            "position": 5,
            "description": "学习LLM的评估方法、基准测试与模型对比。",
            "files": [
                {"name": "evaluation-metrics.md", "title": "评估指标", "id": "evaluation-metrics"},
                {"name": "benchmark-datasets.md", "title": "基准数据集", "id": "benchmark-datasets"},
                {"name": "automated-evaluation.md", "title": "自动化评估", "id": "automated-evaluation"},
                {"name": "human-evaluation.md", "title": "人工评估", "id": "human-evaluation"},
                {"name": "model-comparison.md", "title": "模型对比", "id": "model-comparison"}
            ]
        },
        "llm-rag": {
            "label": "检索增强生成",
            "position": 6,
            "description": "掌握RAG系统的构建、优化与应用。",
            "files": [
                {"name": "rag-fundamentals.md", "title": "RAG基础", "id": "rag-fundamentals"},
                {"name": "retrieval-systems.md", "title": "检索系统", "id": "retrieval-systems"},
                {"name": "vector-databases.md", "title": "向量数据库", "id": "vector-databases"},
                {"name": "chunking-strategies.md", "title": "分块策略", "id": "chunking-strategies"},
                {"name": "embedding-techniques.md", "title": "嵌入技术", "id": "embedding-techniques"},
                {"name": "rag-optimization.md", "title": "RAG优化", "id": "rag-optimization"}
            ]
        },
        "llm-agents": {
            "label": "智能代理与工具使用",
            "position": 7,
            "description": "学习智能代理的设计、工具调用与多代理系统。",
            "files": [
                {"name": "agent-frameworks.md", "title": "代理框架", "id": "agent-frameworks"},
                {"name": "tool-calling.md", "title": "工具调用", "id": "tool-calling"},
                {"name": "planning-reasoning.md", "title": "规划推理", "id": "planning-reasoning"},
                {"name": "multi-agent-systems.md", "title": "多代理系统", "id": "multi-agent-systems"},
                {"name": "agent-evaluation.md", "title": "代理评估", "id": "agent-evaluation"}
            ]
        },
        "multimodal-ai": {
            "label": "多模态AI",
            "position": 8,
            "description": "探索多模态AI模型与跨模态应用。",
            "files": [
                {"name": "vision-language-models.md", "title": "视觉语言模型", "id": "vision-language-models"},
                {"name": "text-to-image.md", "title": "文本到图像", "id": "text-to-image"},
                {"name": "audio-processing.md", "title": "音频处理", "id": "audio-processing"},
                {"name": "video-understanding.md", "title": "视频理解", "id": "video-understanding"},
                {"name": "multimodal-applications.md", "title": "多模态应用", "id": "multimodal-applications"}
            ]
        },
        "llm-development": {
            "label": "LLM开发实践",
            "position": 9,
            "description": "掌握LLM的开发流程、训练技巧与MLOps实践。",
            "files": [
                {"name": "model-training.md", "title": "模型训练", "id": "model-training"},
                {"name": "data-preparation.md", "title": "数据准备", "id": "data-preparation"},
                {"name": "training-strategies.md", "title": "训练策略", "id": "training-strategies"},
                {"name": "model-optimization.md", "title": "模型优化", "id": "model-optimization"},
                {"name": "debugging-techniques.md", "title": "调试技术", "id": "debugging-techniques"},
                {"name": "mlops-for-llm.md", "title": "LLM的MLOps", "id": "mlops-for-llm"}
            ]
        },
        "llm-applications": {
            "label": "领域特定应用",
            "position": 10,
            "description": "探索LLM在各个领域的具体应用与实践案例。",
            "files": [
                {"name": "code-generation.md", "title": "代码生成", "id": "code-generation"},
                {"name": "content-creation.md", "title": "内容创作", "id": "content-creation"},
                {"name": "education-learning.md", "title": "教育学习", "id": "education-learning"},
                {"name": "business-automation.md", "title": "商业自动化", "id": "business-automation"},
                {"name": "scientific-research.md", "title": "科学研究", "id": "scientific-research"},
                {"name": "healthcare-applications.md", "title": "医疗应用", "id": "healthcare-applications"}
            ]
        },
        "llm-research": {
            "label": "前沿研究",
            "position": 11,
            "description": "了解LLM的前沿研究方向与未来发展趋势。",
            "files": [
                {"name": "scaling-laws.md", "title": "缩放定律", "id": "scaling-laws"},
                {"name": "emergent-abilities.md", "title": "涌现能力", "id": "emergent-abilities"},
                {"name": "interpretability.md", "title": "可解释性", "id": "interpretability"},
                {"name": "novel-architectures.md", "title": "新型架构", "id": "novel-architectures"},
                {"name": "efficiency-innovations.md", "title": "效率创新", "id": "efficiency-innovations"},
                {"name": "future-directions.md", "title": "未来方向", "id": "future-directions"}
            ]
        }
    }
    
    # 创建目录和文件
    for dir_name, config in directories_config.items():
        dir_path = base_path / dir_name
        
        # 创建目录
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {dir_path}")
        
        # 创建 _category_.json 文件
        category_config = {
            "label": config["label"],
            "position": config["position"],
            "link": {
                "type": "generated-index",
                "description": config["description"]
            }
        }
        
        category_file = dir_path / "_category_.json"
        with open(category_file, 'w', encoding='utf-8') as f:
            json.dump(category_config, f, ensure_ascii=False, indent=2)
        print(f"创建文件: {category_file}")
        
        # 创建 markdown 文件
        for i, file_info in enumerate(config["files"], 1):
            md_file = dir_path / file_info["name"]
            
            # 创建包含front matter的内容
            content = f"""---
id: {file_info['id']}
sidebar_position: {i}
title: {file_info['title']}
---

# {file_info['title']}

"""
            
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"创建文件: {md_file}")

def main():
    """主函数"""
    print("开始创建LLM知识体系目录结构...")
    print("=" * 50)
    
    try:
        create_directory_structure()
        print("=" * 50)
        print("✅ 目录结构创建完成！")
        print("\n📁 新增的目录包括:")
        print("- llm-security (LLM安全与风险)")
        print("- llm-evaluation (LLM评估与测试)")
        print("- llm-rag (检索增强生成)")
        print("- llm-agents (智能代理与工具使用)")
        print("- multimodal-ai (多模态AI)")
        print("- llm-development (LLM开发实践)")
        print("- llm-applications (领域特定应用)")
        print("- llm-research (前沿研究)")
        print("\n📝 每个文件都包含了正确的front matter格式")
        
    except Exception as e:
        print(f"❌ 创建过程中出现错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 