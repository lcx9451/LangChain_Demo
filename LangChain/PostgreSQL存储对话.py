from langgraph.checkpoint.postgres import PostgresSaver
# 安装 PostgreSQL 检查点器
# pip install langgraph-checkpoint-postgres
import os
import dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool


dotenv.load_dotenv()  #加载当前目录下的 .env 文件

os.environ['OPENAI_API_KEY'] = os.getenv("DASHSCOPE_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("DASHSCOPE_BASE_URL")

# 方法2：分步配置
pgdb_user = os.getenv("PGDB_USER")
pgdb_password = os.getenv("PGDB_PASSWORD")
pgdb_host = os.getenv("PGDB_HOST")
pgdb_port = os.getenv("PGDB_PORT")
pgdb_name = os.getenv("PGDB_NAME")

DB_URI = f"postgresql://{pgdb_user}:{pgdb_password}@{pgdb_host}:{pgdb_port}/{pgdb_name}"
# DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"


@tool
def get_user_info(name: str) -> str:
    """Get user info."""
    return f"用户： {name} 是一个好人."


with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 在 PostgreSQL 中自动创建表
    agent = create_agent(
        "openai:qwen3-max",
        tools=[get_user_info],
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "thread_001"}}

    # result = agent.invoke(
    #     {"messages": [{"role": "user", "content": "你好，我叫张三"}]},
    #     config,
    # )
    # print(result)

    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好，我叫什么名字"}]},
        config,
    )
    print(result1["messages"][-1].content)