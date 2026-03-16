from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
import os
import uuid

# 生产环境配置
# 使用 Postgres 进行持久化存储和检查点

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/deepagents")

# 设置存储
store_ctx = PostgresStore.from_conn_string(DATABASE_URL)
store = store_ctx.__enter__()
store.setup()

# 设置检查点器
checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)
checkpointer.setup()

# 创建后端工厂
def make_backend(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),   # 临时工作区
        routes={
            "/memories/": StoreBackend(runtime),   # 持久记忆
        }
    )

# 创建代理
agent = create_deep_agent(
    model="claude-sonnet-4-6",
    store=store,
    backend=make_backend,
    checkpointer=checkpointer,
    system_prompt="""
你是一名智能助手，具有长期记忆能力。

记忆结构：
- /memories/preferences.txt：用户偏好
- /memories/context.txt：对话上下文
- /memories/knowledge/：学到的知识

在每次对话开始时：
1. 读取 /memories/preferences.txt 了解用户偏好
2. 读取相关的 /memories/knowledge/ 文件

当用户分享新信息时：
1. 更新 /memories/preferences.txt（如果是偏好）
2. 或创建新的 /memories/knowledge/ 文件
"""
)


# 使用示例：跨线程记忆

# 会话 1：保存偏好
thread1 = {"configurable": {"thread_id": str(uuid.uuid4())}}
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "我偏好使用中文交流，请记住。"}]},
    config=thread1,
    version="v2",
)

# 会话 2：不同的线程，但能记住偏好
thread2 = {"configurable": {"thread_id": str(uuid.uuid4())}}
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "你好！"}]},
    config=thread2,
    version="v2",
)
# 代理会用中文回复，因为它从 /memories/preferences.txt 读取了偏好



# 记忆管理工具函数
from langgraph_sdk import get_client
from typing import Optional


class MemoryManager:
    """管理长期记忆的工具类"""

    def __init__(self, deployment_url: str, assistant_id: str):
        self.client = get_client(url=deployment_url)
        self.namespace = (assistant_id, "filesystem")

    async def read_memory(self, path: str) -> Optional[dict]:
        """读取记忆文件（不带 /memories/ 前缀）"""
        try:
            return await self.client.store.get_item(self.namespace, path)
        except Exception:
            return None

    async def write_memory(self, path: str, content: str) -> None:
        """写入记忆文件"""
        from deepagents.backends.utils import create_file_data
        file_data = create_file_data(content)
        await self.client.store.put_item(self.namespace, path, file_data)

    async def delete_memory(self, path: str) -> None:
        """删除记忆文件"""
        await self.client.store.delete_item(self.namespace, path)

    async def list_memories(self, prefix: str = "") -> list:
        """列出记忆文件"""
        items = await self.client.store.search_items(self.namespace)
        if prefix:
            return [item for item in items if item.key.startswith(prefix)]
        return items

    async def prune_old_memories(self, days_old: int = 30) -> None:
        """清理旧的记忆文件"""
        import datetime
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days_old)

        items = await self.list_memories()
        for item in items:
            modified_at = datetime.datetime.fromisoformat(item.value.get("modified_at", ""))
            if modified_at < cutoff:
                await self.delete_memory(item.key)