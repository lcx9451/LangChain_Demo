# LangChain 学习案例

本项目是一个 LangChain 学习案例集合，包含了多种智能体实现和应用场景，旨在帮助开发者快速上手 LangChain 框架并构建各种智能应用。

## 项目结构

```
├── LangChain/           # LangChain 核心示例
│   ├── MySQL查询数据库信息.py
│   ├── PostgreSQL存储对话.py
│   ├── RAG智能体.py
│   ├── SQL智能体.py
│   ├── 上下文感知的客户服务代理.py
│   ├── 人工审核.py
│   ├── 多个护卫.py
│   ├── 多领域创意助手.py
│   ├── 大模型的异步调用.py
│   ├── 客户服务移交示例.py
│   ├── 对话消息处理策略.py
│   ├── 快速构建智能体.py
│   ├── 构建带路由的多源知识库.py
│   ├── 自定义工作流——多来源知识库路由器.py
│   ├── 语义搜索.py
│   ├── 语音智能体.py
│   └── 长期记忆.py
├── DeepAgents/          # 深度智能体示例
│   ├── 构建数据分析智能体.py
│   ├── 生产环境技能配置.py
│   ├── 生产环境沙箱配置.py
│   ├── 生产环境流式传输配置.py
│   ├── 生产环境长期记忆配置.py
│   └── 研究代理.py
├── 智能体集成/           # 智能体集成示例
│   ├── 交接：客户支持.py
│   ├── 子代理：个人助理.py
│   ├── 技能：SQL助手.py
│   └── 路由器：知识库.py
└── .env                 # 环境配置文件
```

## 核心功能

### 1. RAG 智能体
- 实现了基于检索增强生成的智能体
- 支持从网页加载文档并进行向量存储
- 提供上下文相关的回答能力

### 2. SQL 智能体
- 与 SQL 数据库交互
- 支持自动生成 SQL 查询
- 包含人工审核中间件

### 3. 多领域创意助手
- 支持多种创意写作技能
- 包括博客写作、社交媒体内容、电子邮件营销和技术写作
- 可根据需求加载不同技能

### 4. 其他功能
- 语音智能体
- 长期记忆
- 多源知识库
- 客户服务代理
- 异步调用大模型

## 环境配置

1. 安装依赖：
   ```bash
   pip install langchain langchain-core langchain-community langchain-openai langgraph bs4
   ```

2. 配置环境变量：
   - 在 `.env` 文件中设置 API 密钥
   - 或直接在代码中设置环境变量

## 使用示例

### RAG 智能体示例
```python
# 1. 初始化模型和嵌入
model = init_chat_model("gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 2. 加载和处理文档
url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
docs = load_documents(url)
all_splits = split_documents(docs)
vector_store = create_vector_store(all_splits, embeddings)

# 3. 创建 RAG 代理
retrieve_tool = create_retrieve_tool(vector_store)
prompt = "您可以访问从博客文章检索上下文的工具。使用该工具帮助回答用户查询。"
rag_agent = create_rag_agent(model, [retrieve_tool], prompt)

# 4. 查询
query = "What is task decomposition?"
query_agent(rag_agent, query)
```

### SQL 智能体示例
```python
# 1. 初始化模型
model = init_chat_model("gpt-5.2")

# 2. 连接数据库
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# 3. 创建工具包
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# 4. 创建代理
agent = create_agent(model, tools, system_prompt=system_prompt)

# 5. 查询
question = "Which genre on average has the longest tracks?"
for step in agent.stream({"messages": [{"role": "user", "content": question}]}):
    step["messages"][-1].pretty_print()
```

## 学习资源

- [LangChain 官方文档](https://python.langchain.com/docs/get_started/introduction)
- [LangChain 教程](https://python.langchain.com/docs/tutorials/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)

## 许可证

本项目仅供学习和参考使用。