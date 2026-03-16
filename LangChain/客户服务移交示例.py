# 完整客户服务移交示例：展示如何使用移交模式按顺序收集信息并提供支持
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage, AIMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from typing import Callable

# 1. 定义状态类跟踪当前服务步骤
class CustomerSupportState(AgentState):
    """客户服务状态类，用于跟踪整个服务流程的进度和收集的信息"""
    # 当前步骤：从问候开始，依次经过问题识别、保修检查，最后到解决方案
    current_step: str = "greeting"
    # 客户 ID，初始为 None
    customer_id: str | None = None
    # 问题类型，初始为 None
    issue_type: str | None = None
    # 保修状态，初始为 None
    warranty_status: str | None = None

# 2. 定义移交工具：每个工具负责收集特定信息并推进到下一步

@tool
def collect_customer_id(
    customer_id: str,
    runtime: ToolRuntime[None, CustomerSupportState]
) -> Command:
    """收集客户 ID 并转移到问题识别步骤

    Args:
        customer_id: 客户提供的 ID 字符串
        runtime: 工具运行时环境，提供对状态的访问和工具调用 ID

    Returns:
        Command 对象，更新消息、客户 ID 和下一步骤
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Customer ID recorded: {customer_id}",
                    tool_call_id=runtime.tool_call_id
                )
            ],
            "customer_id": customer_id,
            "current_step": "issue_identification"  # 转移到问题识别步骤
        }
    )

@tool
def identify_issue(
    issue_type: str,
    runtime: ToolRuntime[None, CustomerSupportState]
) -> Command:
    """识别问题类型并转移到保修检查步骤

    Args:
        issue_type: 问题类型分类
        runtime: 工具运行时环境

    Returns:
        Command 对象，更新消息、问题类型和下一步骤
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Issue type recorded: {issue_type}",
                    tool_call_id=runtime.tool_call_id
                )
            ],
            "issue_type": issue_type,
            "current_step": "warranty_check"  # 转移到保修检查步骤
        }
    )

@tool
def check_warranty(
    warranty_status: str,
    runtime: ToolRuntime[None, CustomerSupportState]
) -> Command:
    """检查保修状态并转移到解决方案步骤

    Args:
        warranty_status: 保修状态（如在保、过保等）
        runtime: 工具运行时环境

    Returns:
        Command 对象，更新消息、保修状态和下一步骤
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Warranty status: {warranty_status}",
                    tool_call_id=runtime.tool_call_id
                )
            ],
            "warranty_status": warranty_status,
            "current_step": "resolution"  # 转移到解决方案步骤
        }
    )

# 3. 中间件：基于当前步骤动态配置代理行为
@wrap_model_call
def configure_support_step(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据当前客服步骤配置代理行为的中间件函数

    该中间件在每次模型调用前执行，根据当前所处的服务步骤
    动态调整系统提示词和可用工具，确保按正确顺序收集信息

    Args:
        request: 模型请求对象，包含当前状态等信息
        handler: 实际的模型调用处理函数

    Returns:
        处理后的模型响应
    """
    # 从请求状态中获取当前步骤，默认为"greeting"
    step = request.state.get("current_step", "greeting")

    # 定义每个步骤的配置：包括提示词和可用工具
    configs = {
        "greeting": {
            # 问候阶段：问候客户并收集客户 ID
            "prompt": (
                "You are a customer support agent. "
                "Start by greeting the customer and collecting their customer ID. "
                "Use collect_customer_id tool when you have the ID."
            ),
            "tools": [collect_customer_id]  # 仅可用收集客户 ID 工具
        },
        "issue_identification": {
            # 问题识别阶段：询问问题并分类
            "prompt": (
                f"Customer ID: {request.state.get('customer_id')}\n"
                "Ask about their issue and categorize it. "
                "Use identify_issue tool when categorized."
            ),
            "tools": [identify_issue]  # 仅可用识别问题工具
        },
        "warranty_check": {
            # 保修检查阶段：检查产品保修状态
            "prompt": (
                f"Customer ID: {request.state.get('customer_id')}\n"
                f"Issue type: {request.state.get('issue_type')}\n"
                "Check warranty status. Use check_warranty tool."
            ),
            "tools": [check_warranty]  # 仅可用检查保修工具
        },
        "resolution": {
            # 解决方案阶段：基于保修状态提供解决方案
            "prompt": (
                f"Customer ID: {request.state.get('customer_id')}\n"
                f"Issue type: {request.state.get('issue_type')}\n"
                f"Warranty status: {request.state.get('warranty_status')}\n"
                "Provide resolution based on warranty status."
            ),
            "tools": []  # 无需工具，直接提供解决方案
        }
    }

    # 获取当前步骤的配置，如果步骤不存在则使用问候阶段配置
    config = configs.get(step, configs["greeting"])
    # 使用配置覆盖原始请求的系统提示词和工具
    request = request.override(
        system_prompt=config["prompt"],
        tools=config["tools"]
    )
    return handler(request)

# 4. 创建带中间件和检查点的客服代理
support_agent = create_agent(
    model="openai:qwen3-max",  # 使用的模型
    tools=[collect_customer_id, identify_issue, check_warranty],  # 所有可用工具
    state_schema=CustomerSupportState,  # 自定义状态模式
    middleware=[configure_support_step],  # 动态配置中间件
    checkpointer=InMemorySaver()  # 检查点存储器：在轮次间持久化状态
)

# 5. 运行示例：模拟完整的客户服务流程
# 配置对话线程 ID（使用检查点存储器时必需）
config = {"configurable": {"thread_id": "support_001"}}

# 调用代理，传入用户消息和配置
result = support_agent.invoke(
    {"messages": [{"role": "user", "content": "Hello, I need help with my product"}]},
    config=config  # 传入配置以支持状态持久化
)

# 打印所有消息
for msg in result["messages"]:
    msg.pretty_print()
