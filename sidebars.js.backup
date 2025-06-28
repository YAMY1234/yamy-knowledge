// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'LLM基础',
      items: [
        'llm-basics/what-is-llm',
        'llm-basics/llm-principle',
        'llm-basics/use-llm-api',
        'llm-basics/llm-prompt',
        'llm-basics/llm-applications',
      ],
    },
    {
      type: 'category',
      label: 'LLM进阶',
      items: [
        'llm-advanced/finetune-llm',
        'llm-advanced/llm-ethics',
        'llm-advanced/llm-trend',
      ],
    },
    {
      type: 'category',
      label: 'LLM基础设施',
      items: [
        'llm-infra/training-infrastructure',
        'llm-infra/inference-optimization',
        {
          type: 'category',
          label: '推理优化详解',
          items: [
            'llm-infra/inference-optimization/model-compression',
            'llm-infra/inference-optimization/batching-strategies',
            'llm-infra/inference-optimization/hardware-acceleration',
            'llm-infra/inference-optimization/caching-strategies',
            'llm-infra/inference-optimization/pipeline-optimization',
          ],
        },
        'llm-infra/model-deployment',
        'llm-infra/distributed-parallelism',
      ],
    },
  ],

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'LLM基础',
      items: ['llm-basics/what-is-llm'],
    },
    {
      type: 'category',
      label: 'LLM进阶',
      items: ['llm-advanced/finetune-llm'],
    },
    {
      type: 'category',
      label: 'LLM基础设施',
      items: ['llm-infra/training-infrastructure'],
    },
  ],
  */
};

export default sidebars;
