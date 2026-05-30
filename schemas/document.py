from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class UpsertDocumentItem(BaseModel):
    """
    单条文档写入模型。

    这里要求调用方只传业务上真正关心的数据：文档主键、文本、来源和扩展属性。
    embedding 由服务端统一生成，避免客户端各自处理导致维度、模型版本不一致。
    """

    id: str = Field(..., description="业务主键，建议使用稳定且可读的 ID。")
    text: str = Field(..., min_length=1, description="待写入 Milvus 的原始文本内容。")
    source: str = Field(default="default", description="文档来源，例如 运维、开发、实习学习、问题排查。")
    tags: list[str] = Field(default_factory=list, description="标签集合，便于分类和展示。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展业务属性。")


class UpsertDocumentsRequest(BaseModel):
    items: list[UpsertDocumentItem] = Field(..., min_length=1, description="待批量写入或更新的文档列表。")


class UpsertDocumentsResponse(BaseModel):
    collection_name: str
    upserted_count: int
    primary_keys: list[str]


class SearchDocumentsRequest(BaseModel):
    query_text: str = Field(..., min_length=1, description="查询文本，服务端会先将它转成向量。")
    top_k: int = Field(default=5, ge=1, le=50, description="返回结果数量。")
    source: str | None = Field(default=None, description="按 source 做等值过滤。")


class SearchHit(BaseModel):
    id: str
    score: float
    text: str
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SearchDocumentsResponse(BaseModel):
    collection_name: str
    query_text: str
    top_k: int
    hits: list[SearchHit]


class GetDocumentResponse(BaseModel):
    id: str
    text: str
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DeleteDocumentResponse(BaseModel):
    id: str
    deleted: bool
