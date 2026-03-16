# 组合多个 guardrails
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware, hook_config, AgentMiddleware
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.tools import tool
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model
from typing import Any


@tool
def search_tool(query: str) -> str:
    """使用 Google 搜索并返回结果。"""
    return "搜索结果：" + query

@tool
def send_email_tool(email: str) -> str:
    """发送一封邮件到指定邮箱。"""
    return "邮件已发送至" + email

class ContentFilterMiddleware(AgentMiddleware):
    """确定性 guardrail：阻止包含禁止关键词的请求。"""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # 获取第一条用户消息
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        # 检查禁止关键词
        for keyword in self.banned_keywords:
            if keyword in content:
                # 在任何处理之前阻止执行
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
                    }],
                    "jump_to": "end"
                }

        return None


class SafetyGuardrailMiddleware(AgentMiddleware):
    """基于模型的 guardrail：使用 LLM 评估响应安全性。"""

    def __init__(self):
        super().__init__()
        self.safety_model = init_chat_model("gpt-4.1-mini")

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # 获取最终 AI 响应
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # 使用模型评估安全性
        safety_prompt = f"""Evaluate if this response is safe and appropriate.
        Respond with only 'SAFE' or 'UNSAFE'.
        Response: {last_message.content}"""

        result = self.safety_model.invoke([{
            "role": "user",
            "content": safety_prompt
        }])

        if "UNSAFE" in result.content:
            last_message.content = "I cannot provide that response. Please rephrase your request."

        return None


agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, send_email_tool],
    middleware=[
        # 第 1 层：确定性输入过滤器（代理前）
        ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),

        # 第 2 层：PII 保护（模型前后）
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),

        # 第 3 层：敏感工具的人工批准
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),

        # 第 4 层：基于模型的安全检查（代理后）
        SafetyGuardrailMiddleware(),
    ],
)