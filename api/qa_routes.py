"""
RAG 问答路由

提供两个端点：
- POST /api/v1/qa/ask：非流式问答，返回完整回答
- POST /api/v1/qa/ask/stream：流式问答，通过 SSE（Server-Sent Events）逐块返回回答
"""

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from schemas.qa import AskRequest, AskResponse
from services.rag_service import RAGService

# 创建 QA 路由器，前缀 /api/v1/qa
router = APIRouter(prefix="/api/v1/qa", tags=["qa"])


def get_rag_service(request: Request) -> RAGService:
    """
    依赖注入函数：从 app.state 获取 RAG 服务实例。

    与文档路由中的 get_vector_service 模式一致，
    确保整个应用共享同一个 RAG 服务实例。
    """
    return request.app.state.rag_service


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    """
    非流式 RAG 问答端点。

    完整流程：召回 -> 粗排 -> LLM 生成 -> 返回完整回答。
    适用于不需要实时输出的场景，如 API 调用、自动化测试。
    """
    return service.ask(request)


@router.post("/ask/stream")
def ask_stream(
    request: AskRequest,
    service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    """
    流式 RAG 问答端点。

    召回阶段与非流式相同，生成阶段通过 SSE 逐块返回。
    适用于前端实时展示回答的场景，用户体验更好。

    SSE 格式：
    - 每个事件：data: {"content": "文本片段"}
    - 结束标记：data: [DONE]
    """

    def event_generator():
        """SSE 事件生成器，将 LLM 流式输出包装为 SSE 格式。"""
        for chunk in service.ask_stream(request):
            # 每个 chunk 包装为 JSON，确保中文正确编码
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        # 流式结束标记
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",  # 禁用缓存，确保实时传输
            "Connection": "keep-alive",  # 保持长连接
        },
    )
