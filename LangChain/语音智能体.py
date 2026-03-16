"""
LangChain 语音代理 - 完整示例（三明治架构）
"""

# ==================== 1. 导入依赖 ====================
import os
import json
import base64
import time
import asyncio
import contextlib
from uuid import uuid4
from typing import AsyncIterator, Optional

import websockets
from fastapi import FastAPI, WebSocket

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableGenerator


# ==================== 2. 设置环境变量 ====================
def setup_environment():
    """设置必要的环境变量"""
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = input("LangSmith API Key: ")
    os.environ["ASSEMBLYAI_API_KEY"] = input("AssemblyAI API Key: ")
    os.environ["CARTESIA_API_KEY"] = input("Cartesia API Key: ")
    print("环境变量设置完成")


# ==================== 3. STT 客户端 ====================
class AssemblyAISTT:
    """AssemblyAI STT 客户端"""

    def __init__(self, api_key: str | None = None, sample_rate: int = 16000):
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        self.sample_rate = sample_rate
        self._ws: WebSocketClientProtocol | None = None

    async def send_audio(self, audio_chunk: bytes) -> None:
        ws = await self._ensure_connection()
        await ws.send(audio_chunk)

    async def close(self):
        if self._ws:
            await self._ws.close()

    async def _ensure_connection(self) -> WebSocketClientProtocol:
        if self._ws is None:
            url = f"wss://streaming.assemblyai.com/v3/ws?sample_rate={self.sample_rate}&format_turns=true"
            self._ws = await websockets.connect(
                url,
                additional_headers={"Authorization": self.api_key}
            )
        return self._ws


# ==================== 4. TTS 客户端 ====================
class CartesiaTTS:
    """Cartesia TTS 客户端"""

    def __init__(
            self,
            api_key: Optional[str] = None,
            voice_id: str = "f6ff7c0c-e396-40a9-a70b-f7607edb6937",
            model_id: str = "sonic-3",
            sample_rate: int = 24000,
            encoding: str = "pcm_s16le",
    ):
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.encoding = encoding
        self._ws: WebSocketClientProtocol | None = None
        self._context_counter = 0

    def _generate_context_id(self) -> str:
        timestamp = int(time.time() * 1000)
        counter = self._context_counter
        self._context_counter += 1
        return f"ctx_{timestamp}_{counter}"

    async def send_text(self, text: str | None) -> None:
        if not text or not text.strip():
            return
        ws = await self._ensure_connection()
        payload = {
            "model_id": self.model_id,
            "transcript": text,
            "voice": {"mode": "id", "id": self.voice_id},
            "output_format": {
                "container": "raw",
                "encoding": self.encoding,
                "sample_rate": self.sample_rate,
            },
            "language": "en",
            "context_id": self._generate_context_id(),
        }
        await ws.send(json.dumps(payload))

    async def close(self):
        if self._ws:
            await self._ws.close()

    async def _ensure_connection(self) -> WebSocketClientProtocol:
        if self._ws is None:
            url = (
                f"wss://api.cartesia.ai/tts/websocket"
                f"?api_key={self.api_key}&cartesia_version=2024-06-10"
            )
            self._ws = await websockets.connect(url)
        return self._ws


# ==================== 5. 定义代理工具 ====================
@tool
def add_to_order(item: str, quantity: int) -> str:
    """将商品添加到客户的三明治订单。"""
    return f"Added {quantity} x {item} to the order."


@tool
def confirm_order(order_summary: str) -> str:
    """与客户确认最终订单。"""
    return f"Order confirmed: {order_summary}. Sending to kitchen."


# ==================== 6. 创建代理 ====================
def create_voice_agent():
    """创建语音代理"""
    agent = create_agent(
        model="anthropic:claude-haiku-4-5",
        tools=[add_to_order, confirm_order],
        system_prompt="""You are a helpful sandwich shop assistant.
Your goal is to take the user's order. Be concise and friendly.
Do NOT use emojis, special characters, or markdown.
Your responses will be read by a text-to-speech engine.""",
        checkpointer=InMemorySaver(),
    )
    return agent


# ==================== 7. STT 流管道 ====================
async def stt_stream(audio_stream: AsyncIterator[bytes]) -> AsyncIterator[VoiceAgentEvent]:
    """STT 流管道"""
    stt = AssemblyAISTT(sample_rate=16000)

    async def send_audio():
        try:
            async for audio_chunk in audio_stream:
                await stt.send_audio(audio_chunk)
        finally:
            await stt.close()

    send_task = asyncio.create_task(send_audio())

    try:
        async for event in stt.receive_events():
            yield event
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            send_task.cancel()
            await send_task
        await stt.close()


# ==================== 8. 代理流管道 ====================
async def agent_stream(event_stream: AsyncIterator[VoiceAgentEvent]) -> AsyncIterator[VoiceAgentEvent]:
    """代理流管道"""
    agent = create_voice_agent()
    thread_id = str(uuid4())

    async for event in event_stream:
        yield event
        if event.type == "stt_output":
            stream = agent.astream(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )
            async for message, _ in stream:
                if message.text:
                    yield AgentChunkEvent.create(message.text)


# ==================== 9. TTS 流管道 ====================
async def tts_stream(event_stream: AsyncIterator[VoiceAgentEvent]) -> AsyncIterator[VoiceAgentEvent]:
    """TTS 流管道"""
    tts = CartesiaTTS()

    async def process_upstream():
        async for event in event_stream:
            yield event
            if event.type == "agent_chunk":
                await tts.send_text(event.text)

    try:
        async for event in merge_async_iters(process_upstream(), tts.receive_events()):
            yield event
    finally:
        await tts.close()


# ==================== 10. 完整管道 ====================
def create_pipeline():
    """创建完整管道"""
    pipeline = (
            RunnableGenerator(stt_stream)
            | RunnableGenerator(agent_stream)
            | RunnableGenerator(tts_stream)
    )
    return pipeline


# ==================== 11. WebSocket 端点 ====================
app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def websocket_audio_stream():
        while True:
            data = await websocket.receive_bytes()
            yield data

    pipeline = create_pipeline()
    output_stream = pipeline.atransform(websocket_audio_stream())

    async for event in output_stream:
        if event.type == "tts_chunk":
            await websocket.send_bytes(event.audio)


print("语音代理完整代码定义完成")