import uuid
from daytona import Daytona, CreateSandboxFromSnapshotParams
from langchain_daytona import DaytonaSandbox
from langchain_anthropic import ChatAnthropic
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver


# 生产环境沙箱配置
# - 每个 thread_id 使用唯一的沙箱
# - 配置 TTL 自动清理
# - 不将机密注入沙箱

class SandboxManager:
    """管理沙箱生命周期的工具类"""

    def __init__(self):
        self.client = Daytona()
        self.sandboxes = {}  # thread_id -> sandbox 映射

    def get_or_create_sandbox(self, thread_id: str):
        """获取或创建沙箱"""
        if thread_id in self.sandboxes:
            return self.sandboxes[thread_id]

        try:
            sandbox = self.client.find_one(labels={"thread_id": thread_id})
        except Exception:
            params = CreateSandboxFromSnapshotParams(
                labels={"thread_id": thread_id},
                auto_delete_interval=3600,  # 1 小时 TTL
            )
            sandbox = self.client.create(params)

        self.sandboxes[thread_id] = sandbox
        return sandbox

    def cleanup_sandbox(self, thread_id: str):
        """清理沙箱"""
        if thread_id in self.sandboxes:
            try:
                self.client.delete(self.sandboxes[thread_id])
            except Exception:
                pass
            del self.sandboxes[thread_id]


# 使用示例
manager = SandboxManager()
thread_id = str(uuid.uuid4())
sandbox = manager.get_or_create_sandbox(thread_id)
backend = DaytonaSandbox(sandbox=sandbox)

checkpointer = MemorySaver()

agent = create_deep_agent(
    model=ChatAnthropic(model="claude-sonnet-4-20250514"),
    backend=backend,
    checkpointer=checkpointer,
    system_prompt="""
你是一名具有沙箱访问权限的编码助手。

安全指南：
- 所有代码执行都在隔离的沙箱中进行
- 不要尝试访问沙箱外的系统
- 使用沙箱文件系统工具进行文件操作
- 使用 execute 工具运行 shell 命令
"""
)

try:
    result = agent.invoke(
        {"messages": [
            {"role": "user", "content": "创建一个 Python 脚本计算斐波那契数列并运行它"}
        ]},
        config={"configurable": {"thread_id": thread_id}},
        version="v2",
    )
    print(result["messages"][-1].content)
finally:
    # 清理沙箱
    manager.cleanup_sandbox(thread_id)