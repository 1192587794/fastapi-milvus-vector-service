"""
RAG 问答服务

串联 RAG 流水线的四个核心阶段：
1. 召回（Recall）：从向量数据库中检索与用户问题相关的文档
   - 稠密向量召回：用 embedding 模型编码问题，做余弦相似度搜索
   - 稀疏 BM25 召回：用 BM25 算法做关键词匹配（可选）
   - 图谱召回：从知识图谱中检索相关实体和关系（可选）
2. 粗排（Coarse Ranking）：用 RRF（Reciprocal Rank Fusion）融合多路召回结果
3. 精排（Fine Ranking）：用 Cross-Encoder 模型对候选文档逐对打分（可选）
4. 生成（Generation）：将召回的文档作为上下文，调用 LLM 生成回答
"""

import logging
from collections.abc import Iterator
from typing import Any

from core.config import Settings
from db.milvus_client import MilvusManager
from schemas.qa import AskRequest, AskResponse, SourceChunk
from services.session_service import SessionService
from utils.bm25_retriever import BM25Retriever
from utils.ollama_chat import OllamaChatClient
from utils.openai_chat import OpenAIChatClient
from utils.reranker import CrossEncoderReranker

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
        reranker: CrossEncoderReranker | None = None,  # 精排器，None 表示不启用精排
        session_service: SessionService | None = None,  # 会话服务，None 表示不启用服务端历史存储
        graph_retriever: Any | None = None,  # 图谱召回器，None 表示不启用图谱召回
    ) -> None:
        self.settings = settings
        self.milvus_manager = milvus_manager
        self.embedding = embedding
        self.llm_client = llm_client
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.session_service = session_service
        self.graph_retriever = graph_retriever

    def ask(self, request: AskRequest) -> AskResponse:
        """
        非流式 RAG 问答主流程。

        流程：
        1. 加载对话历史（从 Redis 或客户端传入）
        2. 召回 + 粗排 -> sources（相关文档列表）
        3. 计算置信度 -> confidence
        4. 构造 prompt + 调用 LLM -> answer
        5. 保存对话历史到 Redis（如果启用了会话服务）
        6. 组装响应返回
        """
        # --- 阶段 0：会话管理 ---
        session_id, history = self._resolve_session(request)

        # --- 阶段 1：召回 + 粗排 ---
        sources, hybrid_used, rewritten_query = self._recall(
            question=request.question,
            top_k=request.top_k,
            source_filter=request.source,
            history=history,
        )

        # --- 阶段 2：置信度评估 ---
        confidence = self._compute_confidence(sources)

        # --- 阶段 3：生成 ---
        messages = self._build_messages(request.question, sources, history)
        answer = self.llm_client.chat(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # --- 阶段 4：保存对话历史 ---
        self._save_to_session(session_id, request.question, answer)

        # --- 阶段 5：组装响应 ---
        return AskResponse(
            question=request.question,
            rewritten_query=rewritten_query if self.settings.enable_query_rewriting else None,
            answer=answer,
            sources=sources,
            llm_provider=self.settings.llm_provider,
            confidence=confidence,
            hybrid_recall_used=hybrid_used,
            graph_recall_used=self.graph_retriever is not None,
            query_rewriting_used=self.settings.enable_query_rewriting,
            session_id=session_id,
        )

    def ask_stream(self, request: AskRequest) -> Iterator[str]:
        """
        流式 RAG 问答。

        召回步骤与非流式相同，生成步骤改为流式输出。
        返回的文本片段由路由层拼装成 SSE 事件流。
        """
        # --- 阶段 0：会话管理 ---
        session_id, history = self._resolve_session(request)

        # --- 阶段 1：召回 + 粗排 ---
        sources, hybrid_used, rewritten_query = self._recall(
            question=request.question,
            top_k=request.top_k,
            source_filter=request.source,
            history=history,
        )

        # --- 阶段 2：流式生成 ---
        # 流式模式下，需要收集所有 chunk 才能保存完整回答
        messages = self._build_messages(request.question, sources, history)
        collected_answer: list[str] = []
        for chunk in self.llm_client.chat_stream(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            collected_answer.append(chunk)
            yield chunk

        # 流式生成完成后，保存完整的对话历史到 Redis
        full_answer = "".join(collected_answer)
        self._save_to_session(session_id, request.question, full_answer)

    def _resolve_session(self, request: AskRequest) -> tuple[str | None, list[dict]]:
        """
        解析会话，从 Redis 加载或创建新会话。

        流程：
        1. 启用了 SessionService + 传了 session_id → 从 Redis 加载历史
        2. 启用了 SessionService + 没传 session_id → 创建新会话
        3. 未启用 SessionService → 无历史（降级为单轮对话）

        参数:
            request: 问答请求

        返回:
            (session_id, history_messages)
        """
        if not self.session_service:
            # Redis 不可用，降级为无状态单轮对话
            return None, []

        if request.session_id:
            # 已有会话，从 Redis 加载历史
            history = self.session_service.get_history(request.session_id)
            return request.session_id, history
        else:
            # 创建新会话
            session_id = self.session_service.create_session()
            return session_id, []

    def _save_to_session(self, session_id: str | None, question: str, answer: str) -> None:
        """
        将本轮问答保存到 Redis 会话中。

        只有当 session_id 存在且 SessionService 可用时才保存。
        保存 user 消息和 assistant 回答各一条。

        参数:
            session_id: 会话 ID（可能为 None）
            question: 用户问题
            answer: LLM 回答
        """
        if not session_id or not self.session_service:
            return

        try:
            self.session_service.append_message(session_id, "user", question)
            self.session_service.append_message(session_id, "assistant", answer)
        except Exception:
            logger.warning("保存对话历史失败 session_id=%s", session_id, exc_info=True)

    def _rewrite_query(self, question: str, history: list[dict] | None = None) -> str:
        """
        查询改写：用 LLM 将用户的口语化问题改写为更适合检索的形式。

        通过对话历史和知识库主题提供上下文，解决代词指代和语境缺失问题。

        典型场景（有上下文时）：
        - 用户先问"Milvus 是什么"，再问"这玩意儿咋用"
          -> 改写为"如何使用 Milvus 向量数据库"
        - 用户先问"向量数据库的原理"，再问"那个东西出了问题"
          -> 改写为"向量数据库连接异常排查"

        参数:
            question: 用户的原始问题
            history: 对话历史列表，格式为 [{"role": "user/assistant", "content": "..."}]

        返回:
            改写后的问题（如果改写失败则返回原始问题）
        """
        if not self.settings.enable_query_rewriting:
            return question

        try:
            # 构建对话历史上下文
            context_parts = []
            if history:
                # 只取最近 N 轮对话
                max_turns = self.settings.query_rewriting_context_turns
                recent_history = history[-(max_turns * 2):] if max_turns > 0 else []

                if recent_history:
                    context_parts.append("最近的对话历史：\n")
                    for msg in recent_history:
                        role = "用户" if msg["role"] == "user" else "助手"
                        # 截断过长的内容，避免 token 浪费
                        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                        context_parts.append(f"{role}：{content}\n")
                    context_parts.append("\n")

            # 添加知识库主题上下文
            kb_topic = ""
            if self.settings.query_rewriting_kb_topic:
                kb_topic = f"知识库主题：{self.settings.query_rewriting_kb_topic}\n\n"

            context = "".join(context_parts) if context_parts else ""

            # 格式化 prompt
            prompt = self.settings.query_rewriting_prompt.format(
                question=question,
                context=context,
                kb_topic=kb_topic,
            )
            messages = [{"role": "user", "content": prompt}]
            rewritten = self.llm_client.chat(
                messages=messages,
                temperature=0.3,  # 低温度，保证改写稳定性
                max_tokens=200,
            )
            # 清理可能的引号和空白
            rewritten = rewritten.strip().strip('"').strip("'").strip()
            if rewritten:
                logger.info("查询改写：'%s' -> '%s'", question, rewritten)
                return rewritten
            else:
                logger.warning("查询改写返回空结果，使用原始问题")
                return question
        except Exception:
            logger.warning("查询改写失败，使用原始问题", exc_info=True)
            return question

    def _recall(
        self,
        question: str,
        top_k: int,
        source_filter: str | None = None,
        history: list[dict] | None = None,
    ) -> tuple[list[SourceChunk], bool, str]:
        """
        召回阶段总入口。

        根据配置决定走哪种召回模式：
        - 纯向量召回（默认）：只用 embedding 做语义搜索
        - 混合召回 + RRF 粗排：同时执行向量召回和 BM25 召回，用 RRF 融合
        - 图谱召回（可选）：从知识图谱中检索相关实体和关系
        - 查询改写（可选）：在召回前用 LLM 改写问题

        参数:
            question: 用户问题
            top_k: 最终返回的文档数量
            source_filter: 可选的来源过滤
            history: 对话历史，用于查询改写的上下文

        返回:
            (召回结果列表, 是否使用了混合召回, 实际用于检索的问题)
        """
        # --- 阶段 0：查询改写（可选）---
        rewritten_query = self._rewrite_query(question, history)

        # 按倍数多取候选，供后续排序截取
        # 例如 top_k=5, multiplier=4，则召回 20 条候选
        recall_limit = top_k * self.settings.rag_recall_multiplier

        # 构造 Milvus 标量过滤表达式
        filter_expr = ""
        if source_filter:
            escaped = source_filter.replace("\\", "\\\\").replace('"', '\\"')
            filter_expr = f'source == "{escaped}"'

        # --- 向量召回（Dense Recall）：语义匹配 ---
        dense_results = self._dense_recall(rewritten_query, recall_limit, filter_expr)

        # --- BM25 稀疏召回（Sparse Recall，可选）---
        sparse_results = []
        use_hybrid = (
            self.settings.enable_hybrid_recall
            and self.bm25_retriever is not None
        )
        if use_hybrid:
            sparse_results = self.bm25_retriever.retrieve(
                query=rewritten_query,
                top_k=recall_limit,
                filter_expr=filter_expr,
            )

        # --- 图谱召回（Graph Recall，可选）---
        graph_results = []
        use_graph = self.graph_retriever is not None
        if use_graph:
            graph_results = self.graph_retriever.retrieve(
                question=rewritten_query,
                top_k=recall_limit,
                max_hops=self.settings.graph_max_hops,
            )

        # --- RRF 粗排（Coarse Ranking）：融合多路结果 ---
        coarse_top_k = recall_limit if self.reranker else top_k
        if use_hybrid or use_graph:
            sources = self._rrf_fusion(
                dense_results, sparse_results, graph_results, coarse_top_k
            )
        else:
            # 纯向量模式：直接截取
            sources = dense_results[:coarse_top_k]

        # --- 精排（Fine Ranking）：Cross-Encoder 逐对打分（可选）---
        if self.reranker:
            sources = self.reranker.rerank(question, sources, top_k)

        return sources, use_hybrid or use_graph, rewritten_query

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
        graph_results: list[dict],
        top_k: int,
    ) -> list[SourceChunk]:
        """
        RRF（Reciprocal Rank Fusion）粗排算法，支持 2-way 或 3-way 融合。

        核心思想：不看分数的绝对值，只看排名。因为向量召回的 cosine 分数、BM25 分数
        和图谱召回分数量纲不同，无法直接比较。RRF 通过排名来融合，天然解决了这个问题。

        3-way RRF 公式：
            score(d) = w_dense / (k + rank_dense(d))
                     + w_sparse / (k + rank_sparse(d))
                     + w_graph / (k + rank_graph(d))

        其中：
        - w_dense + w_sparse + w_graph = 1
        - k: 常数 60，避免排名第一的文档获得过大权重
        - 未出现在某路结果中的文档，该路排名取 max_rank + 1

        参数:
            dense_results: 向量召回结果
            sparse_results: BM25 召回结果
            graph_results: 图谱召回结果
            top_k: 最终返回数量

        返回:
            按 RRF 分数降序排列的 top_k 条结果
        """
        alpha = self.settings.hybrid_recall_alpha
        graph_weight = self.settings.graph_recall_weight

        # 计算各路权重（确保总和为 1）
        if graph_results:
            # 3-way 融合：dense + sparse + graph
            remaining = 1.0 - graph_weight
            w_dense = alpha * remaining
            w_sparse = (1 - alpha) * remaining
            w_graph = graph_weight
        else:
            # 2-way 融合：dense + sparse（保持原有行为）
            w_dense = alpha
            w_sparse = 1 - alpha
            w_graph = 0.0

        # 建立 doc_id -> 排名 的映射（排名从 1 开始）
        dense_rank: dict[str, int] = {}
        for i, doc in enumerate(dense_results):
            dense_rank[doc.id] = i + 1

        sparse_rank: dict[str, int] = {}
        for i, doc in enumerate(sparse_results):
            sparse_rank[doc["id"]] = i + 1

        graph_rank: dict[str, int] = {}
        for i, doc in enumerate(graph_results):
            graph_rank[doc["id"]] = i + 1

        # 收集所有出现在任一路结果中的文档 id
        all_ids = set(dense_rank.keys()) | set(sparse_rank.keys()) | set(graph_rank.keys())

        # 构建 doc_id -> SourceChunk 的映射
        id_to_doc: dict[str, SourceChunk] = {}
        for doc in dense_results:
            id_to_doc[doc.id] = doc
        for doc in sparse_results:
            if doc["id"] not in id_to_doc:
                id_to_doc[doc["id"]] = SourceChunk(
                    id=doc["id"],
                    text=doc["text"],
                    score=doc.get("bm25_score", 0.0),
                    source=doc.get("source"),
                    metadata=doc.get("metadata", {}),
                )
        for doc in graph_results:
            if doc["id"] not in id_to_doc:
                id_to_doc[doc["id"]] = SourceChunk(
                    id=doc["id"],
                    text=doc["text"],
                    score=doc.get("score", 0.5),
                    source=doc.get("source", "graph"),
                    metadata=doc.get("metadata", {}),
                )

        # 对每个文档计算 RRF 分数
        max_dense = len(dense_results) if dense_results else 0
        max_sparse = len(sparse_results) if sparse_results else 0
        max_graph = len(graph_results) if graph_results else 0
        default_rank = max(max_dense, max_sparse, max_graph) + 1

        scored: list[tuple[str, float]] = []
        for doc_id in all_ids:
            d_rank = dense_rank.get(doc_id, default_rank)
            s_rank = sparse_rank.get(doc_id, default_rank)
            g_rank = graph_rank.get(doc_id, default_rank)
            rrf_score = (
                w_dense / (RRF_K + d_rank)
                + w_sparse / (RRF_K + s_rank)
                + w_graph / (RRF_K + g_rank)
            )
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

    def _truncate_history(self, history: list[dict]) -> list[dict]:
        """
        截断对话历史，只保留最近 N 轮。

        逻辑：
        1. 按 rag_max_history_turns 配置截断，保留最近的消息
        2. 确保截断后的历史从 user 消息开始（不能从 assistant 开始）
           因为 LLM 对话格式要求 user/assistant 交替出现

        参数:
            history: 完整的对话历史列表

        返回:
            截断后的对话历史列表
        """
        if not history:
            return []

        max_turns = self.settings.rag_max_history_turns
        if max_turns <= 0:
            return []

        # 每轮包含 user + assistant 两条消息，最多保留 max_turns * 2 条
        max_messages = max_turns * 2
        truncated = history[-max_messages:] if len(history) > max_messages else history

        # 确保从 user 消息开始：如果截断后第一条是 assistant，就丢掉它
        if truncated and truncated[0].get("role") == "assistant":
            truncated = truncated[1:]

        return truncated

    def _build_messages(
        self,
        question: str,
        sources: list[SourceChunk],
        history: list[dict] | None = None,
    ) -> list[dict]:
        """
        将对话历史和召回结果拼成 LLM 对话的 messages 格式。

        拼装顺序（这是 LLM 能正确理解上下文的关键）：
        1. system prompt：定义 LLM 角色和回答规则（含引用标记要求）
        2. 历史对话：之前的 user/assistant 消息（让 LLM 理解上下文）
        3. 当前问题：参考资料（带 [1][2][3] 编号）+ 图谱上下文 + 用户当前问题

        为什么历史放在参考资料前面？
        - LLM 的注意力机制对开头和结尾的内容更敏感
        - system prompt 放最前面确保角色设定不被冲淡
        - 当前问题放最后面确保 LLM 优先回答当前问题
        - 历史对话放在中间，提供上下文背景

        参数:
            question: 用户当前问题
            sources: 召回的参考文档列表
            history: 对话历史（可选，默认为空）
        """
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 插入对话历史（经过截断处理）
        if history:
            truncated = self._truncate_history(history)
            messages.extend(truncated)

        # 构造当前问题的参考资料
        context_parts: list[str] = []
        graph_context_parts: list[str] = []

        for i, src in enumerate(sources, 1):
            # 检查是否包含图谱上下文
            graph_ctx = src.metadata.get("graph_context", "")
            if graph_ctx:
                graph_context_parts.append(graph_ctx)
            context_parts.append(f"[{i}] {src.text}")

        context = "\n\n".join(context_parts)

        # 如果有图谱上下文，添加到参考资料中
        if graph_context_parts:
            unique_graph_ctx = list(set(graph_context_parts))
            graph_text = "\n".join(unique_graph_ctx)
            user_message = f"参考资料：\n{context}\n\n{graph_text}\n\n用户问题：{question}"
        else:
            user_message = f"参考资料：\n{context}\n\n用户问题：{question}"

        messages.append({"role": "user", "content": user_message})

        return messages
