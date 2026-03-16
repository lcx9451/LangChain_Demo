# 完整示例：上下文感知的客户服务代理
# 导入数据类装饰器，用于定义结构化的数据类
from dataclasses import dataclass
# 导入可选类型注解
from typing import Optional
# 从 langchain.agents 导入创建代理的函数和代理状态类型
from langchain.agents import create_agent, AgentState
# 从 langchain.tools 导入工具装饰器和工具运行时类
from langchain.tools import tool, ToolRuntime
# 从 langchain.agents.middleware 导入动态提示装饰器、模型请求类和模型前中间件装饰器
from langchain.agents.middleware import dynamic_prompt, ModelRequest, before_model
# 从 langgraph.runtime 导入运行时类，用于管理代理执行环境
from langgraph.runtime import Runtime
# 从 langgraph.checkpoint.memory 导入内存检查点保存器，用于实现长期记忆功能
from langgraph.checkpoint.memory import InMemorySaver

# 定义上下文模式：使用数据类定义客户上下文的 schema
@dataclass
class CustomerContext:
    # 客户唯一标识符
    customer_id: str
    # 客户姓名
    customer_name: str
    # 订阅层级，可选值为 "free"(免费)、"premium"(高级)、"enterprise"(企业)
    subscription_tier: str  # "free", "premium", "enterprise"
    # 客户偏好语言，默认为英语
    language: str = "en"

# 工具：查找客户信息 - 获取客户的历史交互记录
@tool
def get_customer_history(runtime: ToolRuntime[CustomerContext]) -> str:
    """获取客户的历史记录。"""
    # 从运行时上下文中获取客户 ID
    customer_id = runtime.context.customer_id

    # 检查是否存在存储系统（用于长期记忆）
    if runtime.store:
        # 从存储中查询客户历史记录，键为 ("customers",) 元组，值为 customer_id
        history = runtime.store.get(("customers",), customer_id)
        # 如果找到历史记录，返回历史内容，否则返回默认消息
        if history:
            return history.value.get("history", "No history found")

    # 如果没有存储系统或未找到记录，返回不可用消息
    return "No history available"

# 工具：检查订阅状态 - 查询客户的订阅层级
@tool
def check_subscription(runtime: ToolRuntime[CustomerContext]) -> str:
    """检查客户的订阅状态。"""
    # 从运行时上下文中获取客户的订阅层级
    tier = runtime.context.subscription_tier
    # 返回格式化的订阅状态信息
    return f"Customer has {tier} subscription"

# 动态提示：根据订阅层级调整行为 - 在模型调用前动态生成提示词
@dynamic_prompt
def subscription_aware_prompt(request: ModelRequest) -> str:
    # 从模型请求中提取运行时上下文
    ctx = request.runtime.context

    # 定义不同订阅层级的指导语字典
    tier_instructions = {
        # 免费层级：提供帮助但不承诺高级功能
        "free": "This customer is on a free tier. Be helpful but don't promise premium features.",
        # 高级层级：可以提供高级功能和优先支持
        "premium": "This customer is a premium subscriber. You can offer premium features and priority support.",
        # 企业层级：提供 VIP 服务并升级复杂问题
        "enterprise": "This is an enterprise customer. Provide white-glove service and escalate complex issues."
    }

    # 构建完整的系统提示词，包含客户信息和订阅层级指导
    return f"""You are a customer support assistant.
    - Customer name: {ctx.customer_name}
    - Language: {ctx.language}
    - {tier_instructions.get(ctx.subscription_tier, '')}
    - Always be polite and professional"""

# 中间件：记录请求 - 在模型调用之前执行的日志记录中间件
@before_model
def log_customer_request(state: AgentState, runtime: Runtime[CustomerContext]) -> dict | None:
    # 打印当前处理的客户信息，包括姓名和 ID，用于调试和审计
    print(f"Handling request for {runtime.context.customer_name} ({runtime.context.customer_id})")
    # 返回 None 表示不修改状态，继续执行
    return None

# 创建代理：配置代理的所有组件
agent = create_agent(
    # 指定使用的模型，这里使用 GPT-4.1
    model="gpt-4.1",
    # 注册工具列表，包含客户历史查询和订阅检查两个工具
    tools=[get_customer_history, check_subscription],
    # 注册中间件列表，包含动态提示和日志记录中间件
    middleware=[subscription_aware_prompt, log_customer_request],
    # 指定上下文 schema，用于类型检查和验证
    context_schema=CustomerContext,
    # 启用检查点保存器，实现跨会话的长期记忆功能
    checkpointer=InMemorySaver()  # 启用长期记忆
)

# 调用代理：执行代理任务
result = agent.invoke(
    # 传入消息列表，包含用户的查询内容
    {"messages": [{"role": "user", "content": "What's my subscription status?"}]},
    # 传入客户上下文对象，包含客户的所有相关信息
    context=CustomerContext(
        # 客户 ID
        customer_id="cust_123",
        # 客户姓名
        customer_name="Alice Johnson",
        # 订阅层级为高级会员
        subscription_tier="premium",
        # 语言设置为英语
        language="en"
    )
)