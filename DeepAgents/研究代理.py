# 完整研究代理示例
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

# 设置 API 密钥
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"

# 定义搜索工具
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """运行网络搜索"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# 定义子代理
research_subagent = {
    "name": "researcher",
    "description": "专门用于深入网络研究",
    "system_prompt": "You are an expert researcher. Search thoroughly and compile detailed findings.",
    "tools": [internet_search],
}

# 定义系统提示
research_instructions = """
You are an expert research assistant. Your job is to conduct thorough research 
and write polished reports. Use subagents for specialized research tasks.

Guidelines:
- Always create a todo list before starting
- Use internet_search for gathering information
- Delegate deep research to the researcher subagent
- Save findings to files for reference
- Write comprehensive, well-structured reports
"""

# 创建代理
checkpointer = MemorySaver()

agent = create_deep_agent(
    model="claude-sonnet-4-6",
    tools=[internet_search],
    subagents=[research_subagent],
    system_prompt=research_instructions,
    checkpointer=checkpointer,
    interrupt_on={
        "write_file": True,  # 文件写入前需要批准
    }
)

# 运行代理
question = "Research and write a report on the latest developments in quantum computing"

result = agent.invoke({
    "messages": [{"role": "user", "content": question}]
})

# 打印响应
print("\n" + "="*50)
print("研究报告:")
print("="*50)
print(result["messages"][-1].content)