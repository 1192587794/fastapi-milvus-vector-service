"""RAG 问答接口的请求与响应模型。"""

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """RAG 问答请求。"""

    question: str = Field(..., min_length=1, description="用户提出的问题。")
    top_k: int = Field(default=5, ge=1, le=50, description="最终送入 LLM 的上下文文档数量。")
    source: str | None = Field(default=None, description="按文档来源过滤。")
    temperature: float = Field(default=0.7, ge=0, le=2, description="LLM 生成温度。")
    max_tokens: int = Field(default=1024, ge=1, le=8192, description="LLM 最大生成 token 数。")


class SourceChunk(BaseModel):
    """召回的参考文档片段。"""

    id: str
    text: str
    score: float
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    """RAG 问答响应。"""

    question: str
    answer: str
    sources: list[SourceChunk]
    llm_provider: str
