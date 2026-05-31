"""
Cross-Encoder 精排模块。

精排是 RAG 流水线中的第三步（召回 -> 粗排 -> 精排 -> 生成）。
与粗排（RRF 排名融合）不同，精排使用 Cross-Encoder 模型对 query 和 document 逐对打分，
能捕捉更细粒度的语义相关性，精度远高于粗排。

Cross-Encoder 的工作方式：
- 将 query 和 document 拼接成一个序列，送入 BERT 类模型
- 模型输出一个相关性分数（logit），越高表示越相关
- 与 Bi-Encoder（向量召回用的 embedding 模型）不同，Cross-Encoder 能看到
  query 和 document 的完整交互信息，精度更高但速度更慢

注意：首次调用时会自动下载模型（约 80MB），后续使用本地缓存。
"""

import logging

from schemas.qa import SourceChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-Encoder 精排器。

    使用 sentence-transformers 的 CrossEncoder 对候选文档重新打分排序。
    采用懒加载设计：模型在首次 rerank() 调用时才加载，避免启动时阻塞。
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """
        初始化精排器。

        参数:
            model_name: Hugging Face 模型名称，默认 ms-marco-MiniLM-L-6-v2
                        这是一个在 MS MARCO 数据集上微调的 6 层 MiniLM 模型，
                        约 80MB，精度和速度的平衡点很好。
        """
        self.model_name = model_name
        # 模型懒加载：首次调用 rerank() 时才初始化
        self._model = None

    def _load_model(self):
        """
        懒加载 Cross-Encoder 模型。

        首次调用时：
        1. 从 sentence_transformers 导入 CrossEncoder 类
        2. 加载指定的预训练模型（自动从 Hugging Face 下载，或使用本地缓存）
        3. 记录日志提示模型已就绪

        后续调用直接复用已加载的模型实例。
        """
        if self._model is not None:
            return

        logger.info("正在加载 Cross-Encoder 精排模型：%s", self.model_name)
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name)
        logger.info("Cross-Encoder 精排模型加载完成")

    def rerank(
        self,
        query: str,
        candidates: list[SourceChunk],
        top_k: int,
    ) -> list[SourceChunk]:
        """
        对候选文档做精排。

        流程：
        1. 加载模型（首次调用时）
        2. 构造 (query, doc.text) 对
        3. 用 Cross-Encoder 模型对每对计算相关性分数
        4. 按分数降序排序，取 top_k 条

        参数:
            query: 用户问题
            candidates: 粗排后的候选文档列表
            top_k: 精排后保留的文档数量

        返回:
            精排后的 top_k 条文档，score 字段更新为 Cross-Encoder 分数
        """
        if not candidates:
            return []

        # 确保模型已加载
        self._load_model()

        # 构造 (query, document) 对
        pairs = [(query, doc.text) for doc in candidates]

        # Cross-Encoder 批量打分，返回每个 pair 的相关性分数（logit）
        scores = self._model.predict(pairs)

        # 将分数映射回候选文档
        scored = list(zip(candidates, scores, strict=True))

        # 按 Cross-Encoder 分数降序排序
        scored.sort(key=lambda x: x[1], reverse=True)

        # 取 top_k，更新 score 字段为 Cross-Encoder 分数
        results: list[SourceChunk] = []
        for doc, ce_score in scored[:top_k]:
            results.append(
                SourceChunk(
                    id=doc.id,
                    text=doc.text,
                    score=float(ce_score),  # 用 Cross-Encoder 分数替换原始分数
                    source=doc.source,
                    metadata=doc.metadata,
                )
            )

        return results
