"""RAG 问答服务：串联召回（Recall）和生成（Generation）。"""

import logging
from collections.abc import Iterator
from typing import Any

from core.config import Settings
from db.milvus_client import MilvusManager
from schemas.qa import AskRequest, AskResponse, SourceChunk
from utils.ollama_chat import OllamaChatClient
from utils.openai_chat import OpenAIChatClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个智能问答助手。请根据以下参考资料回答用户的问题。

规则：
1. 只基于提供的参考资料回答，不要编造信息。
2. 如果参考资料中没有相关信息，请明确告知用户。
3. 回答要简洁准确，必要时可以引用参考资料的编号。
4. 使用与用户问题相同的语言回答。"""


class RAGService:
    """RAG 问答服务，负责召回和生成的编排。"""

    def __init__(
        self,
        settings: Settings,
        milvus_manager: MilvusManager,
        embedding: Any,
        llm_client: OllamaChatClient | OpenAIChatClient,
    ) -> None:
        self.settings = settings
        self.milvus_manager = milvus_manager
        self.embedding = embedding
        self.llm_client = llm_client

    def ask(self, request: AskRequest) -> AskResponse:
        """
        非流式 RAG 问答。

        流程：
        1. 召回（Recall）：将问题编码为向量，在 Milvus 中搜索相关文档
        2. 生成（Generation）：将召回的文档作为上下文，调用 LLM 生成回答
        """
        # --- 召回（Recall）---
        sources = self._recall(
            question=request.question,
            top_k=request.top_k,
            source_filter=request.source,
        )

        # --- 生成（Generation）---
        messages = self._build_messages(request.question, sources)
        answer = self.llm_client.chat(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return AskResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            llm_provider=self.settings.llm_provider,
        )

    def ask_stream(self, request: AskRequest) -> Iterator[str]:
        """
        流式 RAG 问答。

        召回步骤与非流式相同，生成步骤改为流式输出。
        返回 SSE 格式的事件流，由路由层拼装。
        """
        # --- 召回（Recall）---
        sources = self._recall(
            question=request.question,
            top_k=request.top_k,
            source_filter=request.source,
        )

        # --- 生成（Generation，流式）---
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
    ) -> list[SourceChunk]:
        """
        召回阶段：向量相似度搜索。

        按 rag_recall_multiplier 倍数多取候选，再截取 top_k 条作为上下文。
        """
        recall_limit = top_k * self.settings.rag_recall_multiplier
        query_vector = self.embedding.encode(question)

        # 构造过滤表达式
        filter_expr = ""
        if source_filter:
            escaped = source_filter.replace("\\", "\\\\").replace('"', '\\"')
            filter_expr = f'source == "{escaped}"'

        raw_result = self.milvus_manager.client.search(
            collection_name=self.settings.milvus_collection,
            data=[query_vector],
            limit=recall_limit,
            filter=filter_expr,
            output_fields=["text", "source", "metadata"],
        )

        sources: list[SourceChunk] = []
        first_batch = raw_result[0] if raw_result else []
        for item in first_batch[:top_k]:
            entity = item.get("entity", {})
            sources.append(
                SourceChunk(
                    id=str(item.get("id", "")),
                    text=entity.get("text", ""),
                    score=float(item.get("distance", 0.0)),
                    source=entity.get("source"),
                    metadata=entity.get("metadata", {}),
                )
            )

        return sources

    def _build_messages(self, question: str, sources: list[SourceChunk]) -> list[dict]:
        """将召回结果拼成 LLM 对话的 messages 格式。"""
        # 构造参考资料文本
        context_parts: list[str] = []
        for i, src in enumerate(sources, 1):
            context_parts.append(f"[{i}] {src.text}")
        context = "\n\n".join(context_parts)

        user_message = f"参考资料：\n{context}\n\n用户问题：{question}"

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
