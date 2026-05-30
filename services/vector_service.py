from datetime import UTC, datetime

from core.config import Settings
from db.milvus_client import MilvusManager
from schemas.document import (
    DeleteDocumentResponse,
    GetDocumentResponse,
    SearchDocumentsRequest,
    SearchDocumentsResponse,
    SearchHit,
    UpsertDocumentsRequest,
    UpsertDocumentsResponse,
)
from utils.ollama_embedding import OllamaTextEmbedding


class VectorDocumentService:
    """
    文档向量业务服务。

    这层负责把 HTTP 请求数据转成 Milvus 能理解的数据结构，
    同时也负责把 Milvus 的返回结果重新整理成稳定的业务响应模型。
    """

    def __init__(self, settings: Settings, milvus_manager: MilvusManager) -> None:
        self.settings = settings
        self.milvus_manager = milvus_manager
        self.embedding = OllamaTextEmbedding(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
            dimension=settings.milvus_vector_dimension,
        )

    def upsert_documents(self, request: UpsertDocumentsRequest) -> UpsertDocumentsResponse:
        """
        批量写入或更新文档。

        处理流程：
        1. 从请求中取出文本；
        2. 调用 embedding 组件生成向量；
        3. 组装成 Milvus 所需的记录列表；
        4. 调用 upsert 实现"存在则更新，不存在则插入"。
        """
        now = datetime.now(UTC)
        texts = [item.text for item in request.items]
        vectors = self.embedding.batch_encode(texts)

        payload = []
        for item, vector in zip(request.items, vectors, strict=True):
            payload.append(
                {
                    "id": item.id,
                    "embedding": vector,
                    "text": item.text,
                    "source": item.source,
                    "tags": item.tags,
                    "metadata": item.metadata,
                    "updated_at": now.isoformat(),
                    # created_at 在 upsert 场景下可能会被覆盖。
                    # 模板里优先保持实现简单；生产环境若要严格区分首次写入时间，
                    # 可在 upsert 前先查是否存在，再决定是否沿用原值。
                    "created_at": now.isoformat(),
                }
            )

        result = self.milvus_manager.client.upsert(
            collection_name=self.settings.milvus_collection,
            data=payload,
        )

        primary_keys = result.get("ids", [item.id for item in request.items])
        return UpsertDocumentsResponse(
            collection_name=self.settings.milvus_collection,
            upserted_count=len(request.items),
            primary_keys=[str(key) for key in primary_keys],
        )

    def search_documents(self, request: SearchDocumentsRequest) -> SearchDocumentsResponse:
        """
        执行"文本转向量"后的向量检索。

        这里同时演示了 Milvus 的两个核心能力：
        - 向量相似度搜索
        - 标量字段过滤
        """
        query_vector = self.embedding.encode(request.query_text)
        filter_expression = self._build_filter_expression(source=request.source)

        raw_result = self.milvus_manager.client.search(
            collection_name=self.settings.milvus_collection,
            data=[query_vector],
            limit=request.top_k,
            filter=filter_expression,
            output_fields=["text", "source", "tags", "metadata", "created_at", "updated_at"],
        )

        hits: list[SearchHit] = []
        first_batch = raw_result[0] if raw_result else []
        for item in first_batch:
            entity = item.get("entity", {})
            hits.append(
                SearchHit(
                    id=str(item.get("id", "")),
                    score=float(item.get("distance", 0.0)),
                    text=entity.get("text", ""),
                    source=entity.get("source"),
                    tags=entity.get("tags", []),
                    metadata=entity.get("metadata", {}),
                    created_at=self._parse_datetime(entity.get("created_at")),
                    updated_at=self._parse_datetime(entity.get("updated_at")),
                )
            )

        return SearchDocumentsResponse(
            collection_name=self.settings.milvus_collection,
            query_text=request.query_text,
            top_k=request.top_k,
            hits=hits,
        )

    def get_document(self, id: str) -> GetDocumentResponse | None:
        """
        按主键查询单条文档。

        这里使用 query 而不是 search，因为此处需求是精确定位一条记录，
        不需要做向量相似度计算。
        """
        result = self.milvus_manager.client.query(
            collection_name=self.settings.milvus_collection,
            filter=self._quote_equals("id", id),
            output_fields=["id", "text", "source", "tags", "metadata", "created_at", "updated_at"],
            limit=1,
        )

        if not result:
            return None

        item = result[0]
        return GetDocumentResponse(
            id=item["id"],
            text=item.get("text", ""),
            source=item.get("source"),
            tags=item.get("tags", []),
            metadata=item.get("metadata", {}),
            created_at=self._parse_datetime(item.get("created_at")),
            updated_at=self._parse_datetime(item.get("updated_at")),
        )

    def delete_document(self, id: str) -> DeleteDocumentResponse:
        """
        按主键删除文档。

        删除前先查一次，主要是为了给调用方一个更清晰的 deleted 布尔结果，
        也便于业务侧区分"真的删掉了"和"本来就不存在"。
        """
        existing = self.get_document(id)
        if existing is None:
            return DeleteDocumentResponse(id=id, deleted=False)

        self.milvus_manager.client.delete(
            collection_name=self.settings.milvus_collection,
            filter=self._quote_equals("id", id),
        )
        return DeleteDocumentResponse(id=id, deleted=True)

    @staticmethod
    def _build_filter_expression(source: str | None) -> str:
        """
        统一构造过滤表达式。

        当前模板只演示 source 过滤，后续你可以很容易扩展出 tenant_id、kb_id 等表达式拼装逻辑。
        """
        if not source:
            return ""
        return VectorDocumentService._quote_equals("source", source)

    @staticmethod
    def _quote_equals(field_name: str, value: str) -> str:
        """为字符串型过滤条件做简单转义，避免引号破坏表达式。"""
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'{field_name} == "{escaped}"'

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        """把 ISO 时间字符串转回 datetime，便于响应模型保持明确类型。"""
        if not value:
            return None
        return datetime.fromisoformat(value)
