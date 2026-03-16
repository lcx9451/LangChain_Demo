from deepagents import create_deep_agent
from langchain.tools import tool
from langgraph.config import get_stream_writer
import time
import asyncio


# 定义带有进度跟踪的工具
@tool
def search_web(query: str) -> str:
    """搜索网络获取信息。"""
    writer = get_stream_writer()
    writer({"status": "searching", "query": query, "progress": 0})
    time.sleep(0.3)
    writer({"status": "processing", "progress": 50})
    time.sleep(0.3)
    writer({"status": "complete", "progress": 100})
    return f"Search results for '{query}': Found 5 relevant articles."


@tool
def analyze_content(content: str) -> str:
    """分析内容并提取关键信息。"""
    writer = get_stream_writer()
    writer({"status": "analyzing", "progress": 0})
    time.sleep(0.3)
    writer({"status": "extracting", "progress": 50})
    time.sleep(0.3)
    writer({"status": "complete", "progress": 100})
    return f"Analysis complete: Key topics identified in '{content[:50]}...'"


# 创建子代理
researcher_subagent = {
    "name": "researcher",
    "description": "执行深入研究和内容分析",
    "system_prompt": (
        "You are a thorough researcher. "
        "Use search_web to find information and analyze_content to process it. "
        "Provide detailed progress updates during your work."
    ),
    "tools": [search_web, analyze_content],
}

# 创建深度代理
agent = create_deep_agent(
    model="claude-sonnet-4-6",
    system_prompt=(
        "You are a research coordinator. "
        "For any research request, delegate to the researcher subagent using the task tool. "
        "After receiving results, provide a concise summary."
    ),
    subagents=[researcher_subagent],
)


# 流式传输处理类
class StreamProcessor:
    """处理流式事件的实用程序类"""

    def __init__(self):
        self.active_subagents = {}
        self.current_source = ""
        self.token_buffer = ""

    def process_chunk(self, chunk):
        """处理单个流式块"""
        is_subagent = any(s.startswith("tools:") for s in chunk["ns"])
        source = "subagent" if is_subagent else "main"

        if chunk["type"] == "updates":
            self._handle_updates(chunk, source)
        elif chunk["type"] == "messages":
            self._handle_messages(chunk, source)
        elif chunk["type"] == "custom":
            self._handle_custom(chunk, source)

    def _handle_updates(self, chunk, source):
        """处理更新事件"""
        for node_name in chunk["data"]:
            if node_name in {"model_request", "tools"}:
                print(f"[{source}] step: {node_name}")

    def _handle_messages(self, chunk, source):
        """处理消息事件"""
        token, metadata = chunk["data"]
        if token.content:
            if source != self.current_source:
                if self.token_buffer:
                    print()
                print(f"\n[{source}] ", end="")
                self.current_source = source
            print(token.content, end="", flush=True)
            self.token_buffer += token.content

    def _handle_custom(self, chunk, source):
        """处理自定义事件"""
        if self.token_buffer:
            print()
            self.token_buffer = ""
        print(f"[{source}] progress:", chunk["data"])

    def finalize(self):
        """完成流式传输"""
        if self.token_buffer:
            print()


# 运行流式传输
processor = StreamProcessor()

for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "Research the latest developments in renewable energy"}]},
        stream_mode=["updates", "messages", "custom"],
        subgraphs=True,
        version="v2",
):
    processor.process_chunk(chunk)

processor.finalize()