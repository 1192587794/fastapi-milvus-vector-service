"""
RAG 问答服务

串联 RAG 流水线的三个核心阶段：
1. 召回（Recall）：从向量数据库中检索与用户问题相关的文档
   - 稠密向量召回：用 embedding 模型编码问题，做余弦相似度搜索
   - 稀疏 BM25 召回：用 BM25 算法做关键词匹配（可选）
2. 粗排（Coarse Ranking）：用 RRF（Reciprocal Rank Fusion）融合两路召回结果
3. 生成（Generation）：将召回的文档作为上下文，调用 LLM 生成回答

精排（Fine Ranking / Reranker）暂未实现，后续可在此处插入 cross-encoder 模块。
"""

import logging
from collections.abc import Iterator
from typing import Any

from core.config import Settings
from db.milvus_client import MilvusManager
from schemas.qa import AskRequest, AskResponse, SourceChunk
from utils.bm25_retriever import BM25Retriever
from utils.ollama_chat import OllamaChatClient
from utils.openai_chat import OpenAIChatClient

logger = logging.getLogger(__name__)

# 系统提示词：告诉 LLM 它的角色和回答规则
# 关键点：要求 LLM 使用 [1][2][3] 编号标注引用来源
SYSTEM_PROMPT = """你是一个智能问答助手。请根据以下参考资料回答用户的问题。

规则：
1. 只基于提供的参考资料回答，不要编造信息。
2. 如果参考资料中没有相关信息，请明确告知用户。
3. 回答中必须使用 [1][2][3] 这样的编号来标注引用来源。
4. 使用与用户问题相同的语言回答。"""

# RRF（Reciprocal Rank Fusion）常数 k
# RRF 公式：score(d) = 1 / (k + rank(d))
# k=60 是论文推荐值，用于平衡排名差异，避免排名第一的文档获得过大权重
RRF_K = 60


class RAGService:
    """
    RAG 问答服务，负责召回、粗排和生成的编排。

    构造函数接收所有依赖组件，通过配置控制是否启用混合召回。
    这样设计的好处是：默认行为不变（纯向量召回），开启混合召回只需改配置 + 注入 BM25Retriever。
    """

    def __init__(
        self,
        settings: Settings,
        milvus_manager: MilvusManager,
        embedding: Any,  # OllamaTextEmbedding 实例，用于将文本编码为向量
        llm_client: OllamaChatClient | OpenAIChatClient,  # LLM 对话客户端
        bm25_retriever: BM25Retriever | None = None,  # BM25 召回器，None 表示不启用混合召回
    ) -> None:
        self.settings = settings
        self.milvus_manager = milvus_manager
        self.embedding = embedding
        self.llm_client = llm_client
        self.bm25_retriever = bm25_retriever

    def ask(self, request: AskRequest) -> AskResponse:
        """
        非流式 RAG 问答主流程。

        流程：
        1. 召回 + 粗排 -> sources（相关文档列表）
        2. 计算置信度 -> confidence
        3. 构造 prompt + 调用 LLM -> answer
        4. 组装响应返回
        """
        # --- 阶段 1：召回 + 粗排 ---
        sources, hybrid_used = self._recall(
            question=request.question,
            top_k=request.top_k,
            source_filter=request.source,
        )

        # --- 阶段 2：置信度评估 ---
        confidence = self._compute_confidence(sources)

        # --- 阶段 3：生成 ---
        messages = self._build_messages(request.question, sources)
        answer = self.llm_client.chat(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # --- 阶段 4：组装响应 ---
        return AskResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            llm_provider=self.settings.llm_provider,
            confidence=confidence,
            hybrid_recall_used=hybrid_used,
        )

    def ask_stream(self, request: AskRequest) -> Iterator[str]:
        """
        流式 RAG 问答。

        召回步骤与非流式相同，生成步骤改为流式输出。
        返回的文本片段由路由层拼装成 SSE 事件流。
        """
        # --- 阶段 1：召回 + 粗排 ---
        sources, hybrid_used = self._recall(
            question=request.question,
            top_k=request.top_k,
            source_filter=request.source,
        )

        # --- 阶段 2：流式生成 ---
        messages = self._build_messages(request.question, sources)
        for chunk in self.llm_client.chat_stream(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            yield chunk

    def _recall(
        self,
        question: str,
        top_k: int,
        source_filter: str | None = None,
    ) -> tuple[list[SourceChunk], bool]:
        """
        召回阶段总入口。

        根据配置决定走哪种召回模式：
        - 纯向量召回（默认）：只用 embedding 做语义搜索
        - 混合召回 + RRF 粗排：同时执行向量召回和 BM25 召回，用 RRF 融合

        参数:
            question: 用户问题
            top_k: 最终返回的文档数量
            source_filter: 可选的来源过滤

        返回:
            (召回结果列表, 是否使用了混合召回)
        """
        # 按倍数多取候选，供后续排序截取
        # 例如 top_k=5, multiplier=4，则召回 20 条候选
        recall_limit = top_k * self.settings.rag_recall_multiplier

        # 构造 Milvus 标量过滤表达式
        filter_expr = ""
        if source_filter:
            escaped = source_filter.replace("\\", "\\\\").replace('"', '\\"')
            filter_expr = f'source == "{escaped}"'

        # --- 向量召回（Dense Recall）：语义匹配 ---
        dense_results = self._dense_recall(question, recall_limit, filter_expr)

        # --- 判断是否开启混合召回 ---
        use_hybrid = (
            self.settings.enable_hybrid_recall
            and self.bm25_retriever is not None
        )

        if use_hybrid:
            # --- BM25 稀疏召回（Sparse Recall）：关键词匹配 ---
            sparse_results = self.bm25_retriever.retrieve(
                query=question,
                top_k=recall_limit,
                filter_expr=filter_expr,
            )

            # --- RRF 粗排（Coarse Ranking）：融合两路结果 ---
            sources = self._rrf_fusion(dense_results, sparse_results, top_k)
            return sources, True
        else:
            # 纯向量模式：直接截取 top_k
            return dense_results[:top_k], False

    def _dense_recall(
        self,
        question: str,
        limit: int,
        filter_expr: str,
    ) -> list[SourceChunk]:
        """
        稠密向量召回。

        将用户问题通过 embedding 模型编码为向量，然后在 Milvus 中做余弦相似度搜索。
        这是语义级别的匹配，能理解同义词和上下文。
        """
        # 将问题文本编码为向量
        query_vector = self.embedding.encode(question)

        # 在 Milvus 中做向量相似度搜索
        raw_result = self.milvus_manager.client.search(
            collection_name=self.settings.milvus_collection,
            data=[query_vector],
            limit=limit,
            filter=filter_expr,
            output_fields=["text", "source", "metadata"],
        )

        # 将 Milvus 返回的原始结果转换为 SourceChunk 模型
        sources: list[SourceChunk] = []
        first_batch = raw_result[0] if raw_result else []
        for item in first_batch:
            entity = item.get("entity", {})
            sources.append(
                SourceChunk(
                    id=str(item.get("id", "")),
                    text=entity.get("text", ""),
                    score=float(item.get("distance", 0.0)),  # cosine 相似度
                    source=entity.get("source"),
                    metadata=entity.get("metadata", {}),
                )
            )

        return sources

    def _rrf_fusion(
        self,
        dense_results: list[SourceChunk],
        sparse_results: list[dict],
        top_k: int,
    ) -> list[SourceChunk]:
        """
        RRF（Reciprocal Rank Fusion）粗排算法。

        核心思想：不看分数的绝对值，只看排名。因为向量召回的 cosine 分数和 BM25 分数
        量纲不同，无法直接比较。RRF 通过排名来融合，天然解决了这个问题。

        RRF 公式：
            score(d) = alpha / (k + rank_dense(d)) + (1 - alpha) / (k + rank_sparse(d))

        其中：
        - alpha: 融合权重，越大越偏向向量召回（默认 0.5 表示等权重）
        - k: 常数 60，避免排名第一的文档获得过大权重
        - rank_dense(d): 文档 d 在向量召回中的排名（从 1 开始）
        - rank_sparse(d): 文档 d 在 BM25 召回中的排名（从 1 开始）
        - 如果文档 d 只出现在一路结果中，另一路排名取 max_rank + 1

        参数:
            dense_results: 向量召回结果
            sparse_results: BM25 召回结果
            top_k: 最终返回数量

        返回:
            按 RRF 分数降序排列的 top_k 条结果
        """
        alpha = self.settings.hybrid_recall_alpha

        # 建立 doc_id -> 排名 的映射（排名从 1 开始）
        dense_rank: dict[str, int] = {}
        for i, doc in enumerate(dense_results):
            dense_rank[doc.id] = i + 1

        sparse_rank: dict[str, int] = {}
        for i, doc in enumerate(sparse_results):
            sparse_rank[doc["id"]] = i + 1

        # 收集所有出现在任一路结果中的文档 id
        all_ids = set(dense_rank.keys()) | set(sparse_rank.keys())

        # 构建 doc_id -> SourceChunk 的映射
        # 优先使用向量召回的结果（因为包含 cosine score）
        id_to_doc: dict[str, SourceChunk] = {}
        for doc in dense_results:
            id_to_doc[doc.id] = doc
        for doc in sparse_results:
            if doc["id"] not in id_to_doc:
                # 只在 BM25 结果中出现的文档，用 bm25_score 作为初始 score
                id_to_doc[doc["id"]] = SourceChunk(
                    id=doc["id"],
                    text=doc["text"],
                    score=doc.get("bm25_score", 0.0),
                    source=doc.get("source"),
                    metadata=doc.get("metadata", {}),
                )

        # 对每个文档计算 RRF 分数
        # 未出现在某路结果中的文档，该路排名取 max_rank + 1（惩罚）
        default_rank = max(len(dense_results), len(sparse_results)) + 1
        scored: list[tuple[str, float]] = []
        for doc_id in all_ids:
            d_rank = dense_rank.get(doc_id, default_rank)
            s_rank = sparse_rank.get(doc_id, default_rank)
            rrf_score = alpha / (RRF_K + d_rank) + (1 - alpha) / (RRF_K + s_rank)
            scored.append((doc_id, rrf_score))

        # 按 RRF 分数降序排序
        scored.sort(key=lambda x: x[1], reverse=True)

        # 返回 top_k 条结果，用 RRF 分数替换原始分数
        results: list[SourceChunk] = []
        for doc_id, rrf_score in scored[:top_k]:
            doc = id_to_doc[doc_id]
            results.append(
                SourceChunk(
                    id=doc.id,
                    text=doc.text,
                    score=rrf_score,
                    source=doc.source,
                    metadata=doc.metadata,
                )
            )

        return results

    def _compute_confidence(self, sources: list[SourceChunk]) -> float:
        """
        计算回答置信度（0-1）。

        依据 top-1 召回分数来估算：
        - 纯向量模式：直接取 cosine 相似度（0-1），越高越可信
        - 混合召回模式：RRF 分数范围约为 0-0.033，按比例换算到 0-1

        当 top-1 分数低于 0.3 时，说明最相关的文档也不太相关，额外打折。

        注意：这是一个简单的启发式计算，更精确的置信度可以用
        LLM 自身的 logprobs 或者多次采样一致性来评估。
        """
        if not sources:
            return 0.0

        top_score = sources[0].score

        if self.settings.enable_hybrid_recall and self.bm25_retriever:
            # RRF 分数换算
            # 最高可能分数 = 2 / (RRF_K + 1) ≈ 0.0328（当 alpha=0.5，两路排名都是 1）
            max_possible = 2.0 / (RRF_K + 1)
            confidence = min(top_score / max_possible, 1.0)
        else:
            # 向量余弦相似度，范围本身就是 0-1
            confidence = max(0.0, min(top_score, 1.0))

        # top-1 分数太低时额外打折
        if confidence < 0.3:
            confidence *= 0.5

        return round(confidence, 4)

    def _build_messages(self, question: str, sources: list[SourceChunk]) -> list[dict]:
        """
        将召回结果拼成 LLM 对话的 messages 格式。

        构造逻辑：
        1. system prompt：定义 LLM 角色和回答规则（含引用标记要求）
        2. user message：参考资料（带 [1][2][3] 编号）+ 用户问题

        这样 LLM 在回答时会自然地使用 [1][2] 来标注引用来源。
        """
        context_parts: list[str] = []
        for i, src in enumerate(sources, 1):
            context_parts.append(f"[{i}] {src.text}")
        context = "\n\n".join(context_parts)

        user_message = f"参考资料：\n{context}\n\n用户问题：{question}"

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
