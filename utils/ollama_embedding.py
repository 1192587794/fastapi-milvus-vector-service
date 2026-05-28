import httpx


class OllamaTextEmbedding:
    """通过本地 Ollama 服务调用 embedding 模型。"""

    def __init__(self, model: str, base_url: str, dimension: int) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dimension = dimension

    def encode(self, text: str) -> list[float]:
        return self._embed(text)

    def batch_encode(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        resp = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        vectors = data.get("embeddings", [])
        if not vectors:
            raise RuntimeError(f"Ollama returned empty embeddings: {data}")
        return vectors[0]
