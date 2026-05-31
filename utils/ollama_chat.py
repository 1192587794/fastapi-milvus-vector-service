"""
Ollama 对话客户端

调用本地 Ollama 服务的 /api/chat 接口，支持非流式和流式两种模式。
与 utils/ollama_embedding.py 保持相同的 httpx + tenacity 技术栈。
"""

import json
import logging
from typing import Iterator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class OllamaChatClient:
    """通过本地 Ollama 服务调用对话模型。"""

    def __init__(self, model: str, base_url: str, timeout: int = 120) -> None:
        """
        初始化 Ollama 对话客户端。

        参数:
            model: Ollama 中已拉取的模型名称，如 "qwen2.5:7b"
            base_url: Ollama 服务地址，如 "http://localhost:11434"
            timeout: HTTP 请求超时时间（秒），对话生成通常比 embedding 慢，所以默认 120s
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        非流式对话，返回完整回答文本。

        参数:
            messages: 对话历史，格式为 [{"role": "system/user/assistant", "content": "..."}]
            temperature: 生成温度，越高越随机（0-2）
            max_tokens: 最大生成 token 数（Ollama 中对应 num_predict 参数）

        返回:
            LLM 生成的完整回答文本
        """
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,  # 非流式：等待完整响应
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,  # Ollama 用 num_predict 控制最大生成长度
                },
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama 响应格式：{"message": {"role": "assistant", "content": "..."}, "done": true}
        return data.get("message", {}).get("content", "")

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """
        流式对话，逐块 yield 文本片段。

        Ollama 流式响应格式为 NDJSON（每行一个 JSON 对象）：
        {"message": {"role": "assistant", "content": "你"}, "done": false}
        {"message": {"role": "assistant", "content": "好"}, "done": false}
        {"message": {"role": "assistant", "content": ""}, "done": true}

        参数:
            同 chat() 方法

        生成:
            逐块输出 LLM 生成的文本片段
        """
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,  # 流式：逐块接收响应
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            # 逐行读取 NDJSON
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
                # done=true 表示生成结束
                if chunk.get("done"):
                    break
