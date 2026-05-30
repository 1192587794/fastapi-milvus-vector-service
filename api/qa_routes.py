"""RAG 问答路由。"""

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from schemas.qa import AskRequest, AskResponse
from services.rag_service import RAGService

router = APIRouter(prefix="/api/v1/qa", tags=["qa"])


def get_rag_service(request: Request) -> RAGService:
    """从 app.state 获取 RAG 服务实例。"""
    return request.app.state.rag_service


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    """非流式 RAG 问答：召回相关文档，调用 LLM 生成回答。"""
    return service.ask(request)


@router.post("/ask/stream")
def ask_stream(
    request: AskRequest,
    service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    """流式 RAG 问答：召回相关文档，流式输出 LLM 生成的回答。"""

    def event_generator():
        for chunk in service.ask_stream(request):
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
