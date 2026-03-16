# LangChain Learning Examples

This project is a collection of LangChain learning examples, including various agent implementations and application scenarios, designed to help developers quickly get started with the LangChain framework and build various intelligent applications.

## Project Structure

```
├── LangChain/           # LangChain core examples
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
├── DeepAgents/          # Deep agent examples
│   ├── 构建数据分析智能体.py
│   ├── 生产环境技能配置.py
│   ├── 生产环境沙箱配置.py
│   ├── 生产环境流式传输配置.py
│   ├── 生产环境长期记忆配置.py
│   └── 研究代理.py
├── 智能体集成/           # Agent integration examples
│   ├── 交接：客户支持.py
│   ├── 子代理：个人助理.py
│   ├── 技能：SQL助手.py
│   └── 路由器：知识库.py
└── .env                 # Environment configuration file
```

## Core Features

### 1. RAG Agent
- Implements Retrieval-Augmented Generation agent
- Supports loading documents from web pages and vector storage
- Provides context-aware answering capabilities

### 2. SQL Agent
- Interacts with SQL databases
- Supports automatic SQL query generation
- Includes human review middleware

### 3. Multi-domain Creative Assistant
- Supports multiple creative writing skills
- Includes blog writing, social media content, email marketing, and technical writing
- Can load different skills based on needs

### 4. Other Features
- Voice agent
- Long-term memory
- Multi-source knowledge base
- Customer service agent
- Asynchronous LLM calls

## Environment Setup

1. Install dependencies:
   ```bash
   pip install langchain langchain-core langchain-community langchain-openai langgraph bs4
   ```

2. Configure environment variables:
   - Set API keys in the `.env` file
   - Or directly set environment variables in code

## Usage Examples

### RAG Agent Example
```python
# 1. Initialize model and embeddings
model = init_chat_model("gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 2. Load and process documents
url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
docs = load_documents(url)
all_splits = split_documents(docs)
vector_store = create_vector_store(all_splits, embeddings)

# 3. Create RAG agent
retrieve_tool = create_retrieve_tool(vector_store)
prompt = "You can access tools to retrieve context from blog posts. Use this tool to help answer user queries."
rag_agent = create_rag_agent(model, [retrieve_tool], prompt)

# 4. Query
query = "What is task decomposition?"
query_agent(rag_agent, query)
```

### SQL Agent Example
```python
# 1. Initialize model
model = init_chat_model("gpt-5.2")

# 2. Connect to database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# 3. Create toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# 4. Create agent
agent = create_agent(model, tools, system_prompt=system_prompt)

# 5. Query
question = "Which genre on average has the longest tracks?"
for step in agent.stream({"messages": [{"role": "user", "content": question}]}):
    step["messages"][-1].pretty_print()
```

## Learning Resources

- [LangChain Official Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangChain Tutorials](https://python.langchain.com/docs/tutorials/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## License

This project is for learning and reference purposes only.