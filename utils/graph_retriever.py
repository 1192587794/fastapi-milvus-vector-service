"""
图谱召回器模块。

本模块实现了将知识图谱集成到 RAG 流水线中的能力，
作为第三路召回源（与向量召回、BM25 召回并行工作）。

RAG 流水线中的位置：
    用户问题
        |
        v
    +---+---+---+
    |   |   |   |
    v   v   v   v
  Dense Sparse Graph  <-- 三路召回并行执行
    |   |   |
    +---+---+
        |
        v
    RRF 融合排序  <-- 3-way Reciprocal Rank Fusion
        |
        v
    Cross-Encoder 精排（可选）
        |
        v
    LLM 生成回答

图谱召回的工作流程：
1. 从用户问题中提取实体（如"高血压"）
2. 在知识图谱中查找匹配的实体
3. 多跳遍历获取相关实体和关系
4. 收集关联的文档分片 ID
5. 从 Milvus 中获取分片文本
6. 返回格式化的召回结果

图谱召回 vs 向量召回 vs BM25 召回：
+---------------+----------------+----------------+----------------+
| 特性          | 向量召回       | BM25 召回      | 图谱召回       |
+---------------+----------------+----------------+----------------+
| 匹配方式      | 语义相似度     | 关键词匹配     | 图谱结构遍历   |
| 擅长场景      | 同义词、上下文 | 精确术语       | 多跳推理       |
| 举例          | "心脏"匹配"心"| "阿司匹林"精确 | 高血压->头痛   |
|               | "脏病"         | 匹配           | ->感冒的链路   |
+---------------+----------------+----------------+----------------+
| 响应速度      | 快（<100ms）   | 中（100-500ms）| 快（<100ms）   |
| 依赖          | Embedding 模型 | jieba 分词     | 知识图谱       |
+---------------+----------------+----------------+----------------+

为什么需要图谱召回？
1. 多跳推理：向量召回只能找到与问题语义相似的文档，
   但无法推理"A导致B，B是C的症状"这种链式关系
2. 结构化知识：图谱中的实体和关系是结构化的，比纯文本更精确
3. 补充召回：可以找到向量召回和 BM25 召回遗漏的文档
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GraphRetriever:
    """
    图谱召回器。

    在 RAG 流水线中，与向量召回、BM25 召回并行工作，
    通过 3-way RRF 融合算法将三路结果合并排序。

    使用示例：
        # 创建召回器
        retriever = GraphRetriever(graph_service, milvus_manager, settings)

        # 召回相关文档
        chunks = retriever.retrieve(
            question="高血压会导致什么症状？",
            top_k=5,
            max_hops=2
        )

        # 返回结果
        # [
        #     {"id": "doc1::chunk::0", "text": "...", "score": 0.5, ...},
        #     ...
        # ]
    """

    def __init__(self, graph_service, milvus_manager, settings=None):
        """
        初始化图谱召回器。

        Args:
            graph_service: GraphService 实例，用于查询知识图谱
            milvus_manager: MilvusManager 实例，用于获取文档分片文本
            settings: 可选的 Settings 配置对象
        """
        self._graph_service = graph_service
        self._milvus = milvus_manager
        self._settings = settings

    def retrieve(
        self, question: str, top_k: int = 5, max_hops: int = 2
    ) -> list[dict[str, Any]]:
        """
        基于问题从知识图谱中召回相关文档分片。

        这是图谱召回的主入口方法，处理流程：

        1. 实体抽取：从问题中提取实体
           问题："高血压会导致什么症状？"
           实体：[高血压]

        2. 图谱查询：查找匹配的实体并遍历
           高血压 --(causes)--> 头痛
           高血压 --(causes)--> 胸闷

        3. 收集分片 ID：从实体和关系中提取关联的分片 ID
           chunk_ids = ["doc1::chunk::0", "doc1::chunk::3"]

        4. 获取分片文本：从 Milvus 中查询分片内容
           "doc1::chunk::0" -> "患者有高血压病史..."

        5. 附加图谱上下文：将图谱信息附加到分片的 metadata 中
           metadata["graph_context"] = "[高血压] --(causes)--> [头痛]"

        Args:
            question: 用户问题
            top_k: 返回结果数量
            max_hops: 图谱遍历最大跳数

        Returns:
            召回结果列表，每项包含：
            - id: 分片 ID
            - text: 分片文本
            - score: 相关性分数（图谱召回固定为 0.5）
            - source: 文档来源
            - metadata: 元数据（包含 graph_context）
        """
        try:
            # 步骤1-2：查询图谱
            graph_result = self._graph_service.query_graph(
                question, max_hops=max_hops, top_k=top_k
            )

            chunk_ids = graph_result.get("chunk_ids", [])
            if not chunk_ids:
                return []

            # 步骤3：从 Milvus 获取分片文本
            chunks = self._fetch_chunks_from_milvus(chunk_ids[:top_k])

            # 步骤4：附加图谱上下文信息
            entities = graph_result.get("entities", [])
            relations = graph_result.get("relations", [])
            graph_context = self._graph_service.get_relation_context(entities, relations)

            for chunk in chunks:
                # 将图谱上下文存储在 metadata 中
                # 后续在 _build_messages 中会使用这个信息
                chunk["metadata"]["graph_context"] = graph_context
                chunk["metadata"]["source_type"] = "graph"

                # 图谱召回的分数固定为 0.5
                # 实际的分数会在 RRF 融合时根据排名重新计算
                chunk["score"] = 0.5

            return chunks

        except Exception:
            logger.warning("Graph retrieval failed", exc_info=True)
            return []

    def _fetch_chunks_from_milvus(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """
        从 Milvus 中获取分片文本。

        根据分片 ID 列表，批量查询 Milvus 获取分片的完整内容。

        Args:
            chunk_ids: 分片 ID 列表

        Returns:
            分片数据列表，每项包含 id, text, source, metadata
        """
        if not self._milvus or not chunk_ids:
            return []

        try:
            # 使用 Milvus 的 query 接口批量获取分片
            results = self._milvus.client.query(
                collection_name=self._milvus.collection_name,
                filter=f"id in {chunk_ids}",
                output_fields=["id", "text", "source", "metadata"],
                limit=len(chunk_ids),
            )

            # 转换为标准格式
            chunks = []
            for record in results:
                chunks.append({
                    "id": record.get("id", ""),
                    "text": record.get("text", ""),
                    "score": 0.5,  # 初始分数，后续由 RRF 调整
                    "source": record.get("source", "graph"),
                    "metadata": record.get("metadata", {}),
                })
            return chunks

        except Exception:
            logger.warning("Failed to fetch chunks from Milvus", exc_info=True)
            return []
