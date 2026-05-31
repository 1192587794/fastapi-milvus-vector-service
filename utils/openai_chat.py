"""
OpenAI 兼容对话客户端

支持 OpenAI、DeepSeek、硅基流动等所有兼容 OpenAI Chat Completions API 的服务。
使用原生 httpx 调用，不依赖 openai SDK，保持与项目中 Ollama 客户端一致的技术栈。
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


class OpenAIChatClient:
    """通过 OpenAI 兼容 API 调用对话模型。"""

    def __init__(self, model: str, base_url: str, api_key: str, timeout: int = 120) -> None:
        """
        初始化 OpenAI 兼容对话客户端。

        参数:
            model: 模型名称，如 "gpt-4o-mini"、"deepseek-chat"
            base_url: API 基础地址，如 "https://api.openai.com/v1"
            api_key: API 密钥
            timeout: HTTP 请求超时时间（秒）
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        """构造请求头，包含 Bearer Token 认证。"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        非流式对话，返回完整回答文本。

        调用 POST {base_url}/chat/completions，请求体格式：
        {
            "model": "...",
            "messages": [...],
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": false
        }

        参数:
            messages: 对话历史，格式为 [{"role": "system/user/assistant", "content": "..."}]
            temperature: 生成温度（0-2）
            max_tokens: 最大生成 token 数

        返回:
            LLM 生成的完整回答文本
        """
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenAI 响应格式：{"choices": [{"message": {"content": "..."}}]}
        return data["choices"][0]["message"]["content"]

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """
        流式对话，逐块 yield 文本片段。

        OpenAI 流式响应格式为 SSE（Server-Sent Events）：
        data: {"choices": [{"delta": {"content": "你"}}]}
        data: {"choices": [{"delta": {"content": "好"}}]}
        data: [DONE]

        注意：部分 OpenAI 兼容 API 会在某些 chunk 中返回空的 choices 列表
        （如 usage 统计 chunk），需要跳过这些 chunk。

        参数:
            同 chat() 方法

        生成:
            逐块输出 LLM 生成的文本片段
        """
        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                # OpenAI SSE 格式：每行以 "data: " 开头
                if line.startswith("data: "):
                    payload = line[6:]
                    # [DONE] 标记流式响应结束
                    if payload.strip() == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    # 某些兼容 API 会返回空 choices（如 usage 统计 chunk），需要跳过
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    # delta 中包含增量文本
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
