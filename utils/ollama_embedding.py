import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class OllamaTextEmbedding:
    """通过本地 Ollama 服务调用 embedding 模型。"""

    def __init__(self, model: str, base_url: str, dimension: int) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dimension = dimension

    def encode(self, text: str) -> list[float]:
        """单条文本编码，供搜索查询使用。"""
        return self._embed(text)

    def batch_encode(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        批量编码，利用 Ollama 的 list input 支持一次发送多条文本。

        按 batch_size 分批，每批一次 HTTP 请求。
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self._embed_batch(batch)
            all_vectors.extend(vectors)
        return all_vectors

    def _embed(self, text: str) -> list[float]:
        """单条 embedding 请求。"""
        resp = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text, "dimensions": self.dimension},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        vectors = data.get("embeddings", [])
        if not vectors:
            raise RuntimeError(f"Ollama returned empty embeddings: {data}")
        return vectors[0]

    @retry(
        retry=retry_if_exception_type(
            (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=lambda retry_state: logger.warning(
            "Ollama embed batch failed (attempt %d), retrying: %s",
            retry_state.attempt_number,
            retry_state.outcome.exception() if retry_state.outcome else "unknown",
        ),
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量 embedding 请求，带重试。"""
        resp = httpx.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": texts,
                "dimensions": self.dimension,
            },
            timeout=60,
        )
        if resp.status_code >= 500:
            raise httpx.HTTPStatusError(
                f"Server error {resp.status_code}",
                request=resp.request,
                response=resp,
            )
        resp.raise_for_status()
        data = resp.json()
        vectors = data.get("embeddings", [])
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Ollama returned {len(vectors)} embeddings for {len(texts)} inputs"
            )
        return vectors
