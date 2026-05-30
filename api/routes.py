import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from schemas.document import (
    DeleteDocumentResponse,
    GetDocumentResponse,
    SearchDocumentsRequest,
    SearchDocumentsResponse,
    UpsertDocumentsRequest,
    UpsertDocumentsResponse,
)
from services.vector_service import VectorDocumentService
from utils.file_parser import FileParseError

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def get_vector_service(request: Request) -> VectorDocumentService:
    """
    从 FastAPI 的 application state 中获取已经初始化好的 service。

    这样做可以确保整个应用只维护一份共享的 Milvus 管理器和配置对象。
    """
    return request.app.state.vector_service


@router.post("/upsert", response_model=UpsertDocumentsResponse)
def upsert_documents(
    request: UpsertDocumentsRequest,
    service: VectorDocumentService = Depends(get_vector_service),
) -> UpsertDocumentsResponse:
    """批量写入或更新文档。"""
    return service.upsert_documents(request)


@router.post("/upload", response_model=UpsertDocumentsResponse)
async def upload_document(
    file: UploadFile = File(..., description="PDF 或 DOCX 文件。"),
    doc_id: str | None = Form(default=None, description="业务主键，默认取文件名（去扩展名）。"),
    source: str = Form(default="upload", description="文档来源。"),
    tags: str = Form(default="", description="标签，逗号分隔。"),
    metadata: str = Form(default="{}", description="扩展元数据，JSON 字符串。"),
    service: VectorDocumentService = Depends(get_vector_service),
) -> UpsertDocumentsResponse:
    """上传 PDF 或 DOCX 文件，提取文本后走分片→向量化→写入流程。"""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="文件名不能为空。",
        )

    suffix = Path(file.filename).suffix.lower()
    allowed = service.settings.upload_allowed_extensions
    if suffix not in allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"不支持的文件类型: {suffix}，允许: {', '.join(allowed)}",
        )

    content = await file.read()

    max_bytes = service.settings.upload_max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件过大，最大允许 {service.settings.upload_max_file_size_mb}MB。",
        )

    parsed_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    try:
        parsed_metadata = json.loads(metadata) if metadata and metadata != "{}" else {}
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="metadata 必须是合法的 JSON 字符串。",
        )

    try:
        return service.upload_document(
            filename=file.filename,
            content=content,
            doc_id=doc_id,
            source=source,
            tags=parsed_tags,
            metadata=parsed_metadata,
        )
    except FileParseError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )


@router.post("/search", response_model=SearchDocumentsResponse)
def search_documents(
    request: SearchDocumentsRequest,
    service: VectorDocumentService = Depends(get_vector_service),
) -> SearchDocumentsResponse:
    """根据查询文本执行向量检索。"""
    return service.search_documents(request)


@router.get("/{id}", response_model=GetDocumentResponse)
def get_document(
    id: str,
    service: VectorDocumentService = Depends(get_vector_service),
) -> GetDocumentResponse:
    """按文档主键查询单条记录。"""
    document = service.get_document(id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"document not found: {id}",
        )
    return document


@router.delete("/{id}", response_model=DeleteDocumentResponse)
def delete_document(
    id: str,
    service: VectorDocumentService = Depends(get_vector_service),
) -> DeleteDocumentResponse:
    """按文档主键删除记录。"""
    return service.delete_document(id)
