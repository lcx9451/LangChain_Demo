# 人工审核中间件示例
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.tools import tool
import os
import dotenv

dotenv.load_dotenv()  #加载当前目录下的 .env 文件

os.environ['OPENAI_API_KEY'] = os.getenv("DASHSCOPE_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("DASHSCOPE_BASE_URL")

@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return "I found this information."

@tool
def send_email_tool(message: str) -> str:
    """Send an email."""
    return "Email sent."

@tool
def delete_database_tool(database_name: str) -> str:
    """Delete a database."""
    return "Database deleted."



agent = create_agent(
    model="openai:qwen3-max",
    tools=[search_tool, send_email_tool, delete_database_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 需要批准敏感操作
                "send_email_tool": True,
                "delete_database_tool": True,
                # 自动批准安全操作
                "search_tool": False,
            }
        ),
    ],
    # 在中断之间持久化状态
    checkpointer=InMemorySaver(),
)

# 人工审核需要线程 ID 用于持久化
config = {"configurable": {"thread_id": "some_id"}}

# 代理将在执行敏感工具之前暂停并等待批准
result = agent.invoke(
    {"messages": [{"role": "user", "content": "发送一封邮件到公司邮箱:1405164278@qq.com"}]},
    config=config
)
print(result)


# 使用相同的线程 ID 恢复暂停的对话，并批准敏感操作
result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config
)
# print(result)
print(result["messages"][-1].content)