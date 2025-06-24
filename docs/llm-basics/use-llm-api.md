---
id: use-llm-api
sidebar_position: 3
---

# 如何调用主流 LLM API

## 常见 LLM API
- OpenAI GPT(https://platform.openai.com/)
- 百度文心一言
- 阿里通义千问

## API 调用流程
1. 注册账号，获取 API Key
2. 阅读官方文档，了解接口参数
3. 使用 Python 等语言调用

## Python 示例(以 OpenAI GPT 为例)
```python
import openai
openai.api_key = '你的API密钥'
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "你好，介绍一下大语言模型"}]
)
print(response.choices[0].message["content"])
``` 