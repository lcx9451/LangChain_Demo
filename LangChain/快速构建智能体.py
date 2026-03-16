import os
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy

# 1. 定义系统提示词
SYSTEM_PROMPT = """你是一名专业的天气预测专家，回答时要使用双关语。

你可以使用以下两个工具：
- get_weather_for_location：用于查询指定地区的天气
- get_user_location：用于获取用户所在位置

如果用户向你询问天气，务必先确认具体位置。若能从问题中判断用户想查询当前所在地的天气，调用get_user_location工具获取其位置信息。"""

# 2. 定义上下文模式和工具
@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str

@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气情况。"""
    return f"{city}永远阳光明媚！"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户ID获取用户信息。"""
    user_id = runtime.context.user_id
    return "湖南" if user_id == "1" else "北京"


# 3. 配置大模型
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("DASHSCOPE_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("DASHSCOPE_BASE_URL")

model = init_chat_model(
    model="openai:qwen3-max",
    temperature=0
)

# 4. 定义响应格式
@dataclass
class ResponseFormat:
    """智能体的响应数据模式。"""
    punny_response: str
    weather_conditions: str | None = None

# 5. 配置对话记忆
checkpointer = InMemorySaver()

# 6. 创建并运行智能体
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "1"}}
# 第一次调用
response = agent.invoke(
    {"messages": [{"role": "user", "content": "外面的天气怎么样？"}]},
    config=config,
    context=Context(user_id="1")
)
print(response['structured_response'])

# 多轮对话调用
response = agent.invoke(
    {"messages": [{"role": "user", "content": "谢谢！"}]},
    config=config,
    context=Context(user_id="1")
)
print("\n")
print(response['structured_response'])