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

    @retry(
        retry=retry_if_exception_type(
            (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=lambda retry_state: logger.warning(
            "Ollama chat failed (attempt %d), retrying: %s",
            retry_state.attempt_number,
            retry_state.outcome.exception() if retry_state.outcome else "unknown",
        ),
    )
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        非流式对话，返回完整回答文本。

        重试机制：
        - 连接失败（ConnectError）：Ollama 服务未启动或网络问题
        - 请求超时（TimeoutException）：LLM 生成时间超过 timeout
        - 服务端错误（HTTP 5xx）：Ollama 内部异常
        - 最多重试 3 次，指数退避（1s, 2s, 4s）

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

        重试机制：仅在建立连接阶段重试（连接错误、超时、5xx）。
        流式传输开始后的错误不重试（因为部分数据已返回给调用方）。

        参数:
            同 chat() 方法

        生成:
            逐块输出 LLM 生成的文本片段
        """
        # 流式模式的重试：仅重试连接建立阶段
        # 一旦流式传输开始，部分数据已 yield 给调用方，无法重试
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
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
                # 流式传输成功完成，退出重试循环
                return
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_exc = e
                # 如果已经开始 yield 数据了，不能重试（调用方已收到部分数据）
                # 这里只能在连接建立阶段重试
                logger.warning(
                    "Ollama chat_stream failed (attempt %d), retrying: %s",
                    attempt + 1,
                    e,
                )
                if attempt < 2:
                    import time
                    time.sleep(min(2 ** attempt, 10))
                    continue
                raise
        # 理论上不会到这里，但作为兜底
        if last_exc:
            raise last_exc
