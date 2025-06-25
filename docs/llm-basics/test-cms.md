---
title: CMS 功能测试文档
description: 这是一个测试 Decap CMS 功能的文档
tags: [测试, CMS, 功能验证]
sidebar_position: 1
---

## CMS 功能测试文档

这是一个用于测试 Decap CMS 功能的文档。

### 功能特性

- ✅ 在线 Markdown 编辑
- ✅ 自动保存到 GitHub
- ✅ 图片上传支持
- ✅ 实时预览
- ✅ 自动部署

### 使用说明

1. 访问 `/admin/` 页面
2. 选择要编辑的文档集合
3. 创建或编辑文档
4. 保存后自动同步

### 代码示例

```python
def hello_cms()
    print("Hello, Decap CMS!")
    return "Success"
```

### 图片测试

这里可以插入图片(通过 CMS 上传)
<!-- 图片示例(需要先通过 CMS 上传) -->
<!-- ![测试图片](/img/uploads/test-image.jpg) -->

#### 图片上传说明

1. 在 CMS 编辑器中点击图片按钮
2. 选择或拖拽图片文件
3. 图片会自动保存到 `static/img/uploads/` 目录
4. 在文档中使用 `/img/uploads/文件名` 引用图片

### 下一步

- [ ] 配置 GitHub OAuth
- [ ] 设置生产环境
- [ ] 优化用户体验

---

*最后更新:2024年12月* 