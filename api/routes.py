from fastapi import APIRouter, Depends, HTTPException, Request, status

from schemas.document import (
    DeleteDocumentResponse,
    GetDocumentResponse,
    SearchDocumentsRequest,
    SearchDocumentsResponse,
    UpsertDocumentsRequest,
    UpsertDocumentsResponse,
)
from services.vector_service import VectorDocumentService

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


@router.post("/search", response_model=SearchDocumentsResponse)
def search_documents(
    request: SearchDocumentsRequest,
    service: VectorDocumentService = Depends(get_vector_service),
) -> SearchDocumentsResponse:
    """根据查询文本执行向量检索。"""
    return service.search_documents(request)


@router.get("/{doc_id}", response_model=GetDocumentResponse)
def get_document(
    doc_id: str,
    service: VectorDocumentService = Depends(get_vector_service),
) -> GetDocumentResponse:
    """按文档主键查询单条记录。"""
    document = service.get_document(doc_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"document not found: {doc_id}",
        )
    return document


@router.delete("/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(
    doc_id: str,
    service: VectorDocumentService = Depends(get_vector_service),
) -> DeleteDocumentResponse:
    """按文档主键删除记录。"""
    return service.delete_document(doc_id)
