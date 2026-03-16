from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends.utils import create_file_data

# 创建检查点器和存储
checkpointer = MemorySaver()
store = InMemoryStore()

# 定义技能内容
langgraph_skill = """
---
name: langgraph-docs
description: 使用此技能处理与 LangGraph 相关的请求，以获取相关文档，提供准确、最新的指导。
license: MIT
compatibility: 需要互联网访问以获取文档 URL
metadata:
  author: langchain
  version: "1.0"
allowed-tools: fetch_url
---

# langgraph-docs

## Overview

此技能说明如何访问 LangGraph Python 文档以帮助回答问题并指导实现。

## Instructions

### 1. Fetch the Documentation Index

使用 fetch_url 工具读取以下 URL：
https://docs.langchain.com/llms.txt

### 2. Select Relevant Documentation

根据问题，从索引中识别 2-4 个最相关的文档 URL。

### 3. Fetch Selected Documentation

使用 fetch_url 工具读取选定的文档 URL。

### 4. Provide Accurate Guidance

阅读文档后，完成用户的请求。
"""

research_skill = """
---
name: arxiv-search
description: 使用此技能在 arXiv 预印本存储库中搜索研究论文。
license: MIT
compatibility: 需要互联网访问
metadata:
  author: langchain
  version: "1.0"
allowed-tools: web_search
---

# arxiv-search

## Overview

此技能说明如何在 arXiv 上搜索研究论文。

## Instructions

### 1. Formulate Search Query

根据用户问题构建搜索查询。

### 2. Search arXiv

使用 web_search 工具搜索 arXiv。

### 3. Summarize Findings

总结找到的相关论文。
"""

# 创建技能文件
skills_files = {
    "/skills/langgraph-docs/SKILL.md": create_file_data(langgraph_skill),
    "/skills/arxiv_search/SKILL.md": create_file_data(research_skill),
}

# 创建后端工厂
def make_backend(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={"/memories/": StoreBackend(runtime)}
    )

# 创建具有技能的代理
agent = create_deep_agent(
    model="claude-sonnet-4-6",
    store=store,
    backend=make_backend,
    checkpointer=checkpointer,
    skills=["./skills/"],
    system_prompt="""
你是一名智能助手，具有专业技能。

可用技能：
- langgraph-docs: LangGraph 文档查询
- arxiv-search: arXiv 研究论文搜索

当用户询问相关问题时，使用适当的技能提供准确的指导。
"""
)


# 使用示例

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "如何使用 LangGraph 构建多代理系统？"}
        ],
        "files": skills_files
    },
    config={"configurable": {"thread_id": "12345"}},
    version="v2",
)

print(result.value["messages"][-1].content)



# 推荐的技能目录结构

skills_structure = """
skills/
├── langgraph-docs/
│   ├── SKILL.md              # 必需：技能说明和元数据
│   ├── fetch_docs.py         # 可选：辅助脚本
│   └── templates/
│       └── response.md       # 可选：响应模板
│
├── arxiv_search/
│   ├── SKILL.md              # 必需：技能说明和元数据
│   └── arxiv_search.py       # 可选：搜索代码
│
└── code-review/
    ├── SKILL.md              # 必需：技能说明和元数据
    ├── guidelines.md         # 可选：审查指南
    └── checklists/
        ├── python.md         # 可选：Python 检查清单
        └── javascript.md     # 可选：JavaScript 检查清单
"""

print(skills_structure)