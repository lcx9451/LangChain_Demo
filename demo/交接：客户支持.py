"""
客户支持状态机示例

此示例演示状态机模式。单个代理根据 current_step 状态动态更改其行为，
为顺序信息收集创建状态机。
"""
import uuid
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing import Callable, Literal
from typing_extensions import NotRequired
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime

model = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# 定义可能的工作流步骤
SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]


class SupportState(AgentState):
    """客户支持工作流的状态。"""
    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]


@tool
def record_warranty_status(
        status: Literal["in_warranty", "out_of_warranty"],
        runtime: ToolRuntime[None, SupportState],
) -> Command:
    """记录客户的保修状态并过渡到问题分类。"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"保修状态记录为：{status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        }
    )


@tool
def record_issue_type(
        issue_type: Literal["hardware", "software"],
        runtime: ToolRuntime[None, SupportState],
) -> Command:
    """记录问题类型并过渡到解决方案专家。"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"问题类型记录为：{issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )


@tool
def escalate_to_human(reason: str) -> str:
    """将案例升级至人工支持专家。"""
    return f"升级至人工支持。原因：{reason}"


@tool
def provide_solution(solution: str) -> str:
    """为客户提供问题的解决方案。"""
    return f"提供的解决方案：{solution}"


# 将提示定义为常量
WARRANTY_COLLECTOR_PROMPT = """你是一名帮助处理设备问题的客户支持代理。
当前步骤：保修验证
在此步骤中，你需要：
1. 热情地问候客户
2. 询问他们的设备是否在保修期内
3. 使用 record_warranty_status 记录他们的回复并进入下一步
保持对话友好。不要一次问多个问题。"""

ISSUE_CLASSIFIER_PROMPT = """你是一名帮助处理设备问题的客户支持代理。
当前步骤：问题分类
客户信息：保修状态是 {warranty_status}
在此步骤中，你需要：
1. 请客户描述他们的问题
2. 确定是硬件问题（物理损坏、部件破裂）还是软件问题（应用崩溃、性能问题）
3. 使用 record_issue_type 记录分类并进入下一步
如果不清楚，在分类前先问澄清问题。"""

RESOLUTION_SPECIALIST_PROMPT = """你是一名帮助处理设备问题的客户支持代理。
当前步骤：解决方案
客户信息：保修状态是 {warranty_status}，问题类型是 {issue_type}
在此步骤中，你需要：
1. 对于软件问题：使用 provide_solution 提供故障排除步骤
2. 对于硬件问题：
   - 如果在保修期内：使用 provide_solution 解释保修维修流程
   - 如果不在保修期内：使用 escalate_to_human 进行付费维修选项
提供具体且有帮助的解决方案。"""

# 步骤配置：将步骤名称映射到（提示，工具，所需状态）
STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [provide_solution, escalate_to_human],
        "requires": ["warranty_status", "issue_type"],
    },
}


@wrap_model_call
def apply_step_config(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """根据当前步骤配置代理行为。"""
    current_step = request.state.get("current_step", "warranty_collector")
    step_config = STEP_CONFIG[current_step]

    for key in step_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"{key} 必须在到达 {current_step} 之前设置")

    system_prompt = step_config["prompt"].format(**request.state)
    request = request.override(
        system_prompt=system_prompt,
        tools=step_config["tools"],
    )

    return handler(request)


# 从所有步骤配置中收集所有工具
all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
]

# 使用基于步骤的配置和摘要创建代理
agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[
        apply_step_config,
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=("tokens", 4000),
            keep=("messages", 10)
        )
    ],
    checkpointer=InMemorySaver(),
)

# ============================================================================
# 测试工作流
# ============================================================================
if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [HumanMessage("你好，我的手机屏幕碎了")]},
        config
    )
    result = agent.invoke(
        {"messages": [HumanMessage("是的，还在保修期内")]},
        config
    )
    result = agent.invoke(
        {"messages": [HumanMessage("屏幕是因为掉落而物理破裂的")]},
        config
    )
    result = agent.invoke(
        {"messages": [HumanMessage("我该怎么办？")]},
        config
    )

    for msg in result['messages']:
        msg.pretty_print()