"""
个人助手监督器示例
此示例演示多代理系统的工具调用模式。
监督器代理协调专业化子代理（日历和邮件），这些子代理被包装为工具。
"""

# ==================== 1. 导入依赖 ====================
import os
import json
from typing import Any

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# ==================== 2. 设置环境变量 ====================
def setup_environment():
    """设置必要的环境变量"""
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = input("LangSmith API Key: ")
    os.environ["ANTHROPIC_API_KEY"] = input("Anthropic API Key: ")
    print("环境变量设置完成")

# ==================== 3. 初始化模型 ====================
def setup_model():
    """初始化聊天模型"""
    model = init_chat_model("claude-haiku-4-5-20251001")
    print("模型初始化完成")
    return model

# ==================== 4. 定义底层 API 工具 ====================
@tool
def create_calendar_event(
    title: str,
    start_time: str,   # ISO 格式："2024-01-15T14:00:00"
    end_time: str,     # ISO 格式："2024-01-15T15:00:00"
    attendees: list[str],   # 电子邮件地址
    location: str = ""
) -> str:
    """创建日历事件。需要精确的 ISO 日期时间格式。"""
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"

@tool
def send_email(
    to: list[str],      # 电子邮件地址
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """通过电子邮件 API 发送电子邮件。需要正确格式的地址。"""
    return f"Email sent to {', '.join(to)} - Subject: {subject}"

@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,          # ISO 格式："2024-01-15"
    duration_minutes: int
) -> list[str]:
    """检查给定日期内指定参与者的日历可用性。"""
    return ["09:00", "14:00", "16:00"]

# ==================== 5. 创建子代理 ====================
def create_calendar_agent(model):
    """创建日历代理"""
    CALENDAR_AGENT_PROMPT = (
        "You are a calendar scheduling assistant. "
        "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
        "into proper ISO datetime formats. "
        "Use get_available_time_slots to check availability when needed. "
        "Use create_calendar_event to schedule events. "
        "Always confirm what was scheduled in your final response."
    )

    calendar_agent = create_agent(
        model,
        tools=[create_calendar_event, get_available_time_slots],
        system_prompt=CALENDAR_AGENT_PROMPT,
    )
    print("日历代理创建完成")
    return calendar_agent

def create_email_agent(model):
    """创建邮件代理"""
    EMAIL_AGENT_PROMPT = (
        "You are an email assistant. "
        "Compose professional emails based on natural language requests. "
        "Extract recipient information and craft appropriate subject lines and body text. "
        "Use send_email to send the message. "
        "Always confirm what was sent in your final response."
    )

    email_agent = create_agent(
        model,
        tools=[send_email],
        system_prompt=EMAIL_AGENT_PROMPT,
    )
    print("邮件代理创建完成")
    return email_agent

# ==================== 6. 将子代理包装为工具 ====================
def create_supervisor_tools(calendar_agent, email_agent):
    """创建监督器工具"""

    @tool
    def schedule_event(request: str) -> str:
        """使用自然语言安排日历事件。

        当用户想要创建、修改或检查日历约会时使用此工具。
        处理日期/时间解析、可用性检查和事件创建。

        输入：自然语言日程安排请求
        （例如：'meeting with design team next Tuesday at 2pm'）
        """
        result = calendar_agent.invoke({
            "messages": [{"role": "user", "content": request}]
        })
        return result["messages"][-1].text

    @tool
    def manage_email(request: str) -> str:
        """使用自然语言发送电子邮件。

        当用户想要发送通知、提醒或任何电子邮件通信时使用此工具。
        处理收件人提取、主题生成和邮件撰写。

        输入：自然语言邮件请求
        （例如：'send them a reminder about the meeting'）
        """
        result = email_agent.invoke({
            "messages": [{"role": "user", "content": request}]
        })
        return result["messages"][-1].text

    print("监督器工具创建完成")
    return [schedule_event, manage_email]

# ==================== 7. 创建监督器代理 ====================
def create_supervisor_agent(model, tools):
    """创建监督器代理"""
    SUPERVISOR_PROMPT = (
        "You are a helpful personal assistant. "
        "You can schedule calendar events and send emails. "
        "Break down user requests into appropriate tool calls and coordinate the results. "
        "When a request involves multiple actions, use multiple tools in sequence."
    )

    supervisor_agent = create_agent(
        model,
        tools=tools,
        system_prompt=SUPERVISOR_PROMPT,
    )
    print("监督器代理创建完成")
    return supervisor_agent

# ==================== 8. 运行监督器 ====================
def run_supervisor(supervisor_agent, user_request: str):
    """运行监督器"""
    print(f"\n=== 用户请求 ===")
    print(f"请求：{user_request}")
    print("\n=== 监督器响应 ===")

    # for step in supervisor_agent.stream(
    #     {"messages": [{"role": "user", "content": user_request}]}
    # ):
    #     for update in step.values():
    #         for message in update.get("messages", []):
    #             message.pretty_print()

# ==================== 9. 主函数 ====================
def main():
    """主执行流程"""
    print("=" * 60)
    print("LangChain 多代理个人助手")
    print("=" * 60)

    # 设置环境
    # setup_environment()

    # 初始化组件
    # model = setup_model()
    # calendar_agent = create_calendar_agent(model)
    # email_agent = create_email_agent(model)
    # tools = create_supervisor_tools(calendar_agent, email_agent)
    # supervisor_agent = create_supervisor_agent(model, tools)

    # 运行监督器
    # user_request = (
    #     "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
    #     "and send them an email reminder about reviewing the new mockups."
    # )
    # run_supervisor(supervisor_agent, user_request)

    print("\n完成！")

# 运行
# if __name__ == "__main__":
#     main()