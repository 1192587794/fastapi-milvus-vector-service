import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config import Settings
from db.milvus_client import MilvusManager
from schemas.document import (
    DeleteDocumentResponse,
    GetDocumentResponse,
    SearchDocumentsRequest,
    SearchDocumentsResponse,
    SearchHit,
    UpsertDocumentItem,
    UpsertDocumentsRequest,
    UpsertDocumentsResponse,
)
from utils.file_parser import FileTextExtractor
from utils.ollama_embedding import OllamaTextEmbedding
from utils.text_chunker import chunk_text
from utils.text_cleaner import clean_text

logger = logging.getLogger(__name__)


class VectorDocumentService:
    """
    文档向量业务服务。

    这层负责把 HTTP 请求数据转成 Milvus 能理解的数据结构，
    同时也负责把 Milvus 的返回结果重新整理成稳定的业务响应模型。
    """

    def __init__(
        self,
        settings: Settings,
        milvus_manager: MilvusManager,
        graph_service: Any | None = None,
    ) -> None:
        self.settings = settings
        self.milvus_manager = milvus_manager
        self.graph_service = graph_service
        self.embedding = OllamaTextEmbedding(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
            dimension=settings.milvus_vector_dimension,
        )

    def upsert_documents(self, request: UpsertDocumentsRequest) -> UpsertDocumentsResponse:
        """
        批量写入或更新文档。

        处理流程：
        1. 清除每个文档的旧 chunk；
        2. 文本清洗 + 分片；
        3. 批量 embedding；
        4. 组装 payload 并 upsert。
        """
        now = datetime.now(UTC)

        # 0. 清除每个文档的旧 chunk（包括旧的直接记录和分片记录）
        for item in request.items:
            self._delete_chunks_for_doc(item.id)

        # 1. 清洗 + 分片
        all_chunks: list[dict] = []
        for item in request.items:
            cleaned = clean_text(item.text)
            chunks = chunk_text(
                cleaned,
                doc_id=item.id,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            for chunk in chunks:
                chunk["source"] = item.source
                chunk["tags"] = item.tags
                chunk["metadata"] = {
                    **item.metadata,
                    "parent_id": item.id,
                    "chunk_index": chunk["chunk_index"],
                }
            all_chunks.extend(chunks)

        # 过滤掉空文本 chunk
        all_chunks = [c for c in all_chunks if c["text"].strip()]

        if not all_chunks:
            return UpsertDocumentsResponse(
                collection_name=self.settings.milvus_collection,
                upserted_count=0,
                primary_keys=[],
            )

        # 2. 批量 embedding
        texts = [c["text"] for c in all_chunks]
        vectors = self.embedding.batch_encode(texts)

        # 3. 组装 payload 并 upsert
        payload = []
        for chunk, vector in zip(all_chunks, vectors, strict=True):
            payload.append(
                {
                    "id": chunk["id"],
                    "embedding": vector,
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "tags": chunk["tags"],
                    "metadata": chunk["metadata"],
                    "parent_id": chunk["parent_id"],
                    "updated_at": now.isoformat(),
                    "created_at": now.isoformat(),
                }
            )

        result = self.milvus_manager.client.upsert(
            collection_name=self.settings.milvus_collection,
            data=payload,
        )

        primary_keys = result.get("ids", [c["id"] for c in all_chunks])

        # 文档入库成功后，触发知识图谱构建（可选）
        if self.graph_service:
            for item in request.items:
                try:
                    self.graph_service.build_graph_from_document(
                        doc_id=item.id, text=item.text
                    )
                except Exception:
                    logger.warning("Graph building failed for document %s", item.id, exc_info=True)

        return UpsertDocumentsResponse(
            collection_name=self.settings.milvus_collection,
            upserted_count=len(all_chunks),
            primary_keys=[str(key) for key in primary_keys],
        )

    def upload_document(
        self,
        *,
        filename: str,
        content: bytes,
        doc_id: str | None = None,
        source: str = "upload",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UpsertDocumentsResponse:
        """从上传文件中提取文本，然后走标准的清洗→分片→向量化→写入流程。"""
        extractor = FileTextExtractor()
        text = extractor.extract(filename, content)

        if doc_id is None:
            doc_id = Path(filename).stem

        merged_metadata = {
            "original_filename": filename,
            "file_type": Path(filename).suffix.lower(),
            **(metadata or {}),
        }

        request = UpsertDocumentsRequest(
            items=[
                UpsertDocumentItem(
                    id=doc_id,
                    text=text,
                    source=source,
                    tags=tags or [],
                    metadata=merged_metadata,
                )
            ]
        )
        return self.upsert_documents(request)

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
            output_fields=["text", "source", "tags", "metadata", "parent_id", "chunk_index", "created_at", "updated_at"],
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

        文档经过分片后，原始 doc_id 不再直接存储，而是以 chunk 形式存在。
        这里先尝试精确匹配（兼容旧数据），再按 parent_id 查询所有 chunk 并拼接文本。
        """
        # 先尝试精确匹配（兼容未分片的旧数据）
        result = self.milvus_manager.client.query(
            collection_name=self.settings.milvus_collection,
            filter=self._quote_equals("id", id),
            output_fields=["id", "text", "source", "tags", "metadata", "parent_id", "created_at", "updated_at"],
            limit=1,
        )

        if result:
            item = result[0]
            # 如果是旧格式（没有 parent_id 或 parent_id 为空），直接返回
            if not item.get("parent_id"):
                return GetDocumentResponse(
                    id=item["id"],
                    text=item.get("text", ""),
                    source=item.get("source"),
                    tags=item.get("tags", []),
                    metadata=item.get("metadata", {}),
                    created_at=self._parse_datetime(item.get("created_at")),
                    updated_at=self._parse_datetime(item.get("updated_at")),
                )

        # 按 parent_id 查询所有 chunk
        chunks = self.milvus_manager.client.query(
            collection_name=self.settings.milvus_collection,
            filter=self._quote_equals("parent_id", id),
            output_fields=["id", "text", "source", "tags", "metadata", "chunk_index", "created_at", "updated_at"],
            limit=1000,
        )

        if not chunks:
            return None

        # 按 chunk_index 排序后拼接文本
        chunks.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))
        full_text = "".join(c.get("text", "") for c in chunks)
        first = chunks[0]
        return GetDocumentResponse(
            id=id,
            text=full_text,
            source=first.get("source"),
            tags=first.get("tags", []),
            metadata=first.get("metadata", {}),
            created_at=self._parse_datetime(first.get("created_at")),
            updated_at=self._parse_datetime(first.get("updated_at")),
        )

    def delete_document(self, id: str) -> DeleteDocumentResponse:
        """
        按主键删除文档及其所有 chunk。

        删除前先查一次，给调用方一个更清晰的 deleted 布尔结果。
        """
        existing = self.get_document(id)
        if existing is None:
            return DeleteDocumentResponse(id=id, deleted=False)

        # 删除原始记录（兼容旧数据）+ 所有 chunk
        self._delete_chunks_for_doc(id)

        # 同步清理知识图谱数据（可选）
        if self.graph_service:
            try:
                self.graph_service.delete_graph_for_doc(id)
            except Exception:
                logger.warning("Graph cleanup failed for document %s", id, exc_info=True)

        return DeleteDocumentResponse(id=id, deleted=True)

    def _delete_chunks_for_doc(self, doc_id: str) -> None:
        """删除指定文档的所有 chunk 记录。"""
        self.milvus_manager.client.delete(
            collection_name=self.settings.milvus_collection,
            filter=f'{self._quote_equals("parent_id", doc_id)} or {self._quote_equals("id", doc_id)}',
        )

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
