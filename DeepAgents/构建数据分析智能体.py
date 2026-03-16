"""
LangChain DeepAgents 数据分析代理 - 完整示例
"""

# ==================== 1. 导入依赖 ====================
import csv
import io
import uuid
import os
import getpass

from langchain.tools import tool
from slack_sdk import WebClient
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend


# ==================== 2. 设置环境变量 ====================
def setup_environment():
    """设置必要的环境变量"""
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("LangSmith API Key: ")
    os.environ["SLACK_USER_TOKEN"] = getpass.getpass("Slack User Token: ")
    print("环境变量设置完成")


# ==================== 3. 设置后端 ====================
def setup_backend():
    """设置本地 Shell 后端"""
    backend = LocalShellBackend(
        root_dir=".",
        env={"PATH": "/usr/bin:/bin"}
    )
    # 验证后端
    result = backend.execute("echo ready")
    print(f"后端就绪：{result.output}")
    return backend


# ==================== 4. 创建示例数据 ====================
def create_sample_data(backend):
    """创建并上传示例销售数据"""
    data = [
        ["Date", "Product", "Units Sold", "Revenue"],
        ["2025-08-01", "Widget A", 10, 250],
        ["2025-08-02", "Widget B", 5, 125],
        ["2025-08-03", "Widget A", 7, 175],
        ["2025-08-04", "Widget C", 3, 90],
        ["2025-08-05", "Widget B", 8, 200],
    ]

    text_buf = io.StringIO()
    writer = csv.writer(text_buf)
    writer.writerows(data)
    csv_bytes = text_buf.getvalue().encode("utf-8")
    text_buf.close()

    # 上传到后端
    backend.upload_files([("/home/daytona/data/sales_data.csv", csv_bytes)])
    print("示例数据上传完成")


# ==================== 5. 定义自定义工具 ====================
def create_slack_tool(backend):
    """创建 Slack 发送消息工具"""
    slack_token = os.environ["SLACK_USER_TOKEN"]
    slack_client = WebClient(token=slack_token)
    channel = "C0123456ABC"  # 替换为您的频道

    @tool(parse_docstring=True)
    def slack_send_message(text: str, file_path: str | None = None) -> str:
        """发送消息，可选包括附件（如图片）。

        Args:
            text: (str) 消息文本内容
            file_path: (str) 文件系统中附件的文件路径
        """
        if not file_path:
            slack_client.chat_postMessage(channel=channel, text=text)
        else:
            fp = backend.download_files([file_path])
            slack_client.files_upload_v2(
                channel=channel,
                content=fp[0].content,
                initial_comment=text,
            )
        return "Message sent."

    return slack_send_message


# ==================== 6. 创建代理 ====================
def create_agent(backend, tools):
    """创建 deep agent"""
    checkpointer = InMemorySaver()

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-5",
        tools=tools,
        backend=backend,
        checkpointer=checkpointer,
    )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"代理创建完成，线程 ID: {thread_id}")
    return agent, config


# ==================== 7. 运行代理 ====================
def run_analysis(agent, config):
    """运行数据分析"""
    input_message = {
        "role": "user",
        "content": (
            "Analyze ./data/sales_data.csv in the current dir and generate a beautiful plot. "
            "When finished, send your analysis and the plot to Slack using the tool."
        ),
    }

    print("开始数据分析...")
    for step in agent.stream(
            {"messages": [input_message]},
            config,
            stream_mode="updates",
    ):
        for _, update in step.items():
            if update and (messages := update.get("messages")) and isinstance(messages, list):
                for message in messages:
                    message.pretty_print()


# ==================== 8. 主函数 ====================
def main():
    """主执行流程"""
    print("=" * 60)
    print("LangChain DeepAgents 数据分析代理")
    print("=" * 60)

    # 设置环境
    # setup_environment()

    # 设置后端
    backend = setup_backend()

    # 创建示例数据
    # create_sample_data(backend)

    # 创建工具
    slack_tool = create_slack_tool(backend)

    # 创建代理
    agent, config = create_agent(backend, [slack_tool])

    # 运行分析
    # run_analysis(agent, config)

    print("\n完成！")

# 运行
# if __name__ == "__main__":
#     main()