# 搜索记忆存储
from langgraph.store.memory import InMemoryStore

def embed(texts: list[str]) -> list[list[float]]:
    # 替换为实际的嵌入函数或 LangChain 嵌入对象
    return [[1.0, 2.0] * len(texts)]

# InMemoryStore 将数据保存到内存字典中。生产环境中请使用数据库支持的存储。
store = InMemoryStore(index={"embed": embed, "dims": 2})

user_id = "my-user"
application_context = "chitchat"
namespace = (user_id, application_context)

# 存储记忆
store.put(
    namespace,
    "a-memory",
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
        "my-key": "my-value",
    },
)

# 按 ID 获取记忆
item = store.get(namespace, "a-memory")

# 在此命名空间内搜索记忆，按内容等价性过滤，按向量相似度排序
items = store.search(
    namespace,
    filter={"my-key": "my-value"},
    query="language preferences"
)












# 在工具中读取长期记忆
from dataclasses import dataclass
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

# InMemoryStore 将数据保存到内存字典中。生产环境中请使用数据库支持的存储。
store = InMemoryStore()

# 使用 put 方法将示例数据写入存储
store.put(
    ("users",),  # 命名空间，将相关数据分组在一起（用户命名空间用于用户数据）
    "user_123",  # 命名空间内的键（用户 ID 作为键）
    {"name": "John Smith", "language": "English"},  # 为给定用户存储的数据
)

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """查找用户信息。"""
    # 访问存储 - 与提供给 create_agent 的存储相同
    store = runtime.store
    user_id = runtime.context.user_id
    # 从存储中检索数据 - 返回包含值和元数据的 StoreValue 对象
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# 创建代理
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_user_info],
    # 将存储传递给代理 - 使代理在运行工具时能够访问存储
    store=store,
    context_schema=Context
)

# 运行代理
agent.invoke(
    {"messages": [{"role": "user", "content": "look up user information"}]},
    context=Context(user_id="user_123")
)













# 从工具写入长期记忆
from dataclasses import dataclass
from typing_extensions import TypedDict
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

# InMemoryStore 将数据保存到内存字典中。生产环境中请使用数据库支持的存储。
store = InMemoryStore()

@dataclass
class Context:
    user_id: str

# TypedDict 定义用户信息的结构供 LLM 使用
class UserInfo(TypedDict):
    name: str

# 允许代理更新用户信息的工具（对聊天应用程序很有用）
@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """保存用户信息。"""
    # 访问存储 - 与提供给 create_agent 的存储相同
    store = runtime.store
    user_id = runtime.context.user_id
    # 将数据存储到存储中（命名空间、键、数据）
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

# 创建代理
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[save_user_info],
    store=store,
    context_schema=Context
)

# 运行代理
agent.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith"}]},
    # user_id 在上下文中传递以识别正在更新谁的信息
    context=Context(user_id="user_123")
)

# 您可以直接访问存储以获取值
store.get(("users",), "user_123").value