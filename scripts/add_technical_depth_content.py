#!/usr/bin/env python3
"""
添加技术深度LLM基础设施内容的脚本
专注于底层实现技术：CUDA编程、推理引擎、量化技术等
"""

import json
from pathlib import Path

def create_technical_depth_content():
    """创建技术深度的LLM基础设施内容"""
    
    # 基础路径
    base_path = Path("docs/llm-infra")
    
    # 技术深度目录结构定义
    technical_directories = {
        "gpu-programming": {
            "label": "GPU编程与CUDA优化",
            "position": 5,
            "description": "深入GPU编程、CUDA内核优化与性能调优技术。",
            "files": [
                {"name": "cuda-fundamentals.md", "title": "CUDA编程基础", "id": "cuda-fundamentals"},
                {"name": "cuda-kernel-optimization.md", "title": "CUDA内核优化", "id": "cuda-kernel-optimization"},
                {"name": "cuda-graph-optimization.md", "title": "CUDA图优化", "id": "cuda-graph-optimization"},
                {"name": "memory-hierarchy-optimization.md", "title": "内存层次优化", "id": "memory-hierarchy-optimization"},
                {"name": "warp-level-optimization.md", "title": "Warp级别优化", "id": "warp-level-optimization"},
                {"name": "profiling-debugging.md", "title": "性能分析与调试", "id": "profiling-debugging"}
            ]
        },
        "inference-engines": {
            "label": "推理引擎与编译器",
            "position": 6,
            "description": "探索推理引擎的底层实现与编译器优化技术。",
            "files": [
                {"name": "triton-compiler.md", "title": "Triton编译器", "id": "triton-compiler"},
                {"name": "tvm-tensor-compiler.md", "title": "TVM张量编译器", "id": "tvm-tensor-compiler"},
                {"name": "tensorrt-engine-optimization.md", "title": "TensorRT引擎优化", "id": "tensorrt-engine-optimization"},
                {"name": "custom-operators.md", "title": "自定义算子", "id": "custom-operators"},
                {"name": "graph-optimization.md", "title": "计算图优化", "id": "graph-optimization"},
                {"name": "jit-compilation.md", "title": "即时编译技术", "id": "jit-compilation"}
            ]
        },
        "attention-optimization": {
            "label": "注意力机制优化",
            "position": 7,
            "description": "深入注意力机制的底层实现与优化技术。",
            "files": [
                {"name": "flashattention-internals.md", "title": "FlashAttention原理与实现", "id": "flashattention-internals"},
                {"name": "flashinfer-optimization.md", "title": "FlashInfer优化技术", "id": "flashinfer-optimization"},
                {"name": "custom-attention-kernels.md", "title": "自定义注意力内核", "id": "custom-attention-kernels"},
                {"name": "sparse-attention.md", "title": "稀疏注意力", "id": "sparse-attention"},
                {"name": "linear-attention.md", "title": "线性注意力", "id": "linear-attention"},
                {"name": "attention-variants.md", "title": "注意力变体实现", "id": "attention-variants"}
            ]
        },
        "quantization-compression": {
            "label": "量化与压缩技术",
            "position": 8,
            "description": "深度量化技术与模型压缩的底层实现。",
            "files": [
                {"name": "advanced-quantization.md", "title": "高级量化技术", "id": "advanced-quantization"},
                {"name": "int4-fp8-implementation.md", "title": "INT4/FP8实现", "id": "int4-fp8-implementation"},
                {"name": "dynamic-quantization.md", "title": "动态量化", "id": "dynamic-quantization"},
                {"name": "calibration-techniques.md", "title": "校准技术", "id": "calibration-techniques"},
                {"name": "weight-only-quantization.md", "title": "仅权重量化", "id": "weight-only-quantization"},
                {"name": "activation-quantization.md", "title": "激活量化", "id": "activation-quantization"}
            ]
        },
        "memory-io-optimization": {
            "label": "内存与I/O优化",
            "position": 9,
            "description": "系统级内存管理与I/O优化技术。",
            "files": [
                {"name": "memory-pool-management.md", "title": "内存池管理", "id": "memory-pool-management"},
                {"name": "zero-copy-optimization.md", "title": "零拷贝优化", "id": "zero-copy-optimization"},
                {"name": "prefetching-strategies.md", "title": "预取策略", "id": "prefetching-strategies"},
                {"name": "memory-mapping.md", "title": "内存映射", "id": "memory-mapping"},
                {"name": "numa-optimization.md", "title": "NUMA优化", "id": "numa-optimization"},
                {"name": "storage-optimization.md", "title": "存储优化", "id": "storage-optimization"}
            ]
        },
        "system-optimization": {
            "label": "系统级推理优化",
            "position": 10,
            "description": "推理系统的调度器设计与系统级优化技术。",
            "files": [
                {"name": "scheduler-design.md", "title": "调度器设计", "id": "scheduler-design"},
                {"name": "request-batching-internals.md", "title": "请求批处理内部实现", "id": "request-batching-internals"},
                {"name": "async-execution.md", "title": "异步执行", "id": "async-execution"},
                {"name": "multi-stream-processing.md", "title": "多流处理", "id": "multi-stream-processing"},
                {"name": "load-balancing-algorithms.md", "title": "负载均衡算法", "id": "load-balancing-algorithms"},
                {"name": "resource-management.md", "title": "资源管理", "id": "resource-management"}
            ]
        }
    }
    
    print("开始创建技术深度的LLM基础设施内容...")
    print("=" * 60)
    
    # 创建目录和文件
    for dir_name, config in technical_directories.items():
        dir_path = base_path / dir_name
        
        # 创建目录
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建目录: {dir_path}")
        
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
        print(f"⚙️  创建文件: {category_file}")
        
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

## 概述

本文档深入探讨{file_info['title']}的技术实现细节、优化策略和最佳实践。

## 核心技术

待补充...

## 实现细节

待补充...

## 性能优化

待补充...

## 实践案例

待补充...

## 参考资料

待补充...
"""
            
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"📄 创建文件: {md_file}")

def main():
    """主函数"""
    print("🚀 开始补充技术深度的LLM基础设施内容...")
    print("🔧 专注于底层实现技术：CUDA编程、推理引擎、量化技术等")
    print()
    
    try:
        create_technical_depth_content()
        print("=" * 60)
        print("✅ 技术深度内容创建完成！")
        print()
        print("📚 新增的技术深度目录包括:")
        print("├── 🚀 gpu-programming/ (GPU编程与CUDA优化)")
        print("│   ├── CUDA编程基础")
        print("│   ├── CUDA内核优化") 
        print("│   ├── CUDA图优化")
        print("│   ├── 内存层次优化")
        print("│   ├── Warp级别优化")
        print("│   └── 性能分析与调试")
        print("├── ⚙️  inference-engines/ (推理引擎与编译器)")
        print("│   ├── Triton编译器")
        print("│   ├── TVM张量编译器")
        print("│   ├── TensorRT引擎优化")
        print("│   ├── 自定义算子")
        print("│   ├── 计算图优化")
        print("│   └── 即时编译技术")
        print("├── 🎯 attention-optimization/ (注意力机制优化)")
        print("│   ├── FlashAttention原理与实现")
        print("│   ├── FlashInfer优化技术")
        print("│   ├── 自定义注意力内核")
        print("│   ├── 稀疏注意力")
        print("│   ├── 线性注意力")
        print("│   └── 注意力变体实现")
        print("├── 📦 quantization-compression/ (量化与压缩技术)")
        print("│   ├── 高级量化技术")
        print("│   ├── INT4/FP8实现")
        print("│   ├── 动态量化")
        print("│   ├── 校准技术")
        print("│   ├── 仅权重量化")
        print("│   └── 激活量化")
        print("├── 💾 memory-io-optimization/ (内存与I/O优化)")
        print("│   ├── 内存池管理")
        print("│   ├── 零拷贝优化")
        print("│   ├── 预取策略")
        print("│   ├── 内存映射")
        print("│   ├── NUMA优化")
        print("│   └── 存储优化")
        print("└── 🏗️  system-optimization/ (系统级推理优化)")
        print("    ├── 调度器设计")
        print("    ├── 请求批处理内部实现")
        print("    ├── 异步执行")
        print("    ├── 多流处理")
        print("    ├── 负载均衡算法")
        print("    └── 资源管理")
        print()
        print("🔬 这些内容覆盖了你提到的技术栈：")
        print("✅ Triton (OpenAI编译器)")
        print("✅ TVM张量编译器") 
        print("✅ CUDA编程与内核优化")
        print("✅ CUDA Graph优化")
        print("✅ FlashAttention/FlashInfer深度解析")
        print("✅ 高级量化技术(INT4/FP8)")
        print("✅ 自定义算子与注意力内核")
        print("✅ 推理引擎底层实现")
        
    except Exception as e:
        print(f"❌ 创建过程中出现错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 