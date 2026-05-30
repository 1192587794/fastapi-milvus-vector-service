from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.qa_routes import router as qa_router
from api.routes import router as document_router
from core.config import get_settings
from db.milvus_client import MilvusManager
from services.rag_service import RAGService
from services.vector_service import VectorDocumentService
from utils.llm_factory import create_llm_client

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    在应用启动阶段完成基础资源初始化。

    这里主要做三件事：
    1. 读取统一配置；
    2. 建立 Milvus 客户端并检查集合；
    3. 组装业务 service，挂到 app.state 供路由复用。
    """
    milvus_manager = MilvusManager(settings)
    milvus_manager.ensure_collection()
    app.state.settings = settings
    app.state.milvus_manager = milvus_manager
    vector_service = VectorDocumentService(settings, milvus_manager)
    app.state.vector_service = vector_service

    llm_client = create_llm_client(settings)
    app.state.rag_service = RAGService(
        settings, milvus_manager, vector_service.embedding, llm_client
    )

    yield


app = FastAPI(
    title=settings.app_name,
    debug=settings.app_debug,
    version="0.1.0",
    description="A standardized FastAPI service template for Milvus vector database usage.",
    lifespan=lifespan,
)
app.include_router(document_router)
app.include_router(qa_router)


@app.get("/health")
def health() -> dict:
    """
    基础健康检查接口。

    除了返回 service 状态，也顺手返回当前集合名与连接地址，
    便于你在调试时快速确认服务到底连到了哪个 Milvus 实例。
    """
    info = app.state.milvus_manager.describe_collection()
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "milvus_uri": settings.resolved_milvus_uri,
        "collection_name": settings.milvus_collection,
        "collection_info": info,
    }
