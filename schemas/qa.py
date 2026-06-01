"""
RAG 问答接口的请求与响应模型。

定义了 QA 端点的数据结构，由 FastAPI 自动用于请求校验和响应序列化。
"""

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """
    RAG 问答请求模型。

    对应 POST /api/v1/qa/ask 和 POST /api/v1/qa/ask/stream 的请求体。

    多轮对话用法：
    - 首次提问不传 session_id，服务端创建新会话并在响应中返回 session_id
    - 后续追问传入 session_id，服务端自动从 Redis 加载历史
    - 客户端只需记住 session_id，无需管理历史
    """

    question: str = Field(..., min_length=1, description="用户提出的问题。")
    top_k: int = Field(default=5, ge=1, le=50, description="最终送入 LLM 的上下文文档数量。")
    source: str | None = Field(default=None, description="按文档来源过滤，如 'upload'、'default'。")
    temperature: float = Field(default=0.7, ge=0, le=2, description="LLM 生成温度，越高越随机。")
    max_tokens: int = Field(default=1024, ge=1, le=8192, description="LLM 最大生成 token 数。")
    session_id: str | None = Field(
        default=None,
        description="会话 ID。首次提问不传，后续追问传入上次返回的 session_id。",
    )


class SourceChunk(BaseModel):
    """
    召回的参考文档片段。

    每个 SourceChunk 对应从 Milvus 中检索出的一条文档（或文档分片）。
    score 字段在不同模式下含义不同：
    - 纯向量召回：cosine 相似度（0-1）
    - 混合召回 + RRF：RRF 融合分数（约 0-0.033）
    """

    id: str  # 文档 ID（可能是 chunk ID，如 "doc1::chunk::0"）
    text: str  # 文档文本内容
    score: float  # 相关性分数
    source: str | None = None  # 文档来源
    metadata: dict[str, Any] = Field(default_factory=dict)  # 扩展元数据


class AskResponse(BaseModel):
    """
    RAG 问答响应模型。

    包含 LLM 生成的回答、引用的参考文档、置信度等信息。
    """

    question: str  # 原始问题
    answer: str  # LLM 生成的回答（可能包含 [1][2] 引用标记）
    sources: list[SourceChunk]  # 召回的参考文档列表
    llm_provider: str  # 使用的 LLM 提供商（ollama 或 openai）
    confidence: float = Field(default=0.0, description="回答置信度，0-1 之间，越高越可信。")
    hybrid_recall_used: bool = Field(default=False, description="是否使用了混合召回（向量 + BM25）。")
    graph_recall_used: bool = Field(default=False, description="是否使用了知识图谱召回。")
    session_id: str | None = Field(
        default=None,
        description="会话 ID。首次提问时由服务端生成，后续追问时原样传入即可。",
    )
