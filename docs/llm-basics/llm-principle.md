---
id: llm-principle
title: LLM 的基本原理
sidebar_position: 2
---

## LLM 的基本原理

### Transformer 架构
- LLM 多采用 Transformer 架构，具备强大的序列建模能力
- 关键机制:自注意力(Self-Attention)、多头机制、位置编码

### 预训练与微调
- 预训练:在大规模文本上无监督训练，学习通用语言能力
- 微调:在特定任务/领域数据上有监督训练，提升特定能力

### 训练流程简介
1. 数据收集与清洗
2. 预训练(大规模语料)
3. 微调(小规模高质量数据)
4. 部署与推理 