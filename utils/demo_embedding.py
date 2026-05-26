import hashlib
import math


class DemoTextEmbedding:
    """
    一个轻量、可直接运行的示例文本向量器。

    设计目标不是追求最强语义效果，而是提供一个“零额外模型依赖”的可运行实现，
    让整个 Milvus 接入流程在本地就能完整跑通。

    生产环境中，你应当把这里替换为真正的 embedding 模型或模型服务，例如：
    - OpenAI / Azure OpenAI embedding
    - BGE / M3E / e5 / sentence-transformers
    - 公司内部统一模型服务
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self.dimension = dimension

    def encode(self, text: str) -> list[float]:
        """
        将文本编码为固定维度向量。

        实现思路：
        - 先对文本按字符做稳定哈希；
        - 再把哈希结果映射到固定维度桶；
        - 最后做 L2 归一化，让向量长度稳定，便于使用余弦相似度。

        这种做法适合演示“文本 -> 向量 -> Milvus 检索”的完整链路，
        但不等价于真正的语义模型效果。
        """
        vector = [0.0] * self.dimension
        normalized_text = text.strip()

        if not normalized_text:
            return vector

        for index, char in enumerate(normalized_text):
            digest = hashlib.sha256(f"{index}:{char}".encode("utf-8")).digest()
            bucket = digest[0] % self.dimension
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            magnitude = (digest[2] / 255.0) + 0.1
            vector[bucket] += sign * magnitude

        return self._l2_normalize(vector)

    def batch_encode(self, texts: list[str]) -> list[list[float]]:
        """批量编码接口，便于批量写入时减少上层循环复杂度。"""
        return [self.encode(text) for text in texts]

    @staticmethod
    def _l2_normalize(vector: list[float]) -> list[float]:
        """
        执行 L2 归一化。

        余弦相似度检索通常希望输入向量尺度更稳定，因此这里做归一化处理。
        如果向量全零，则直接返回原向量，避免除零错误。
        """
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
