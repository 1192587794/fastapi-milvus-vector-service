"""
FastAPI 应用入口。

负责：
1. 读取全局配置
2. 在 lifespan 中初始化所有服务组件
3. 注册路由
4. 定义健康检查端点
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.qa_routes import router as qa_router
from api.routes import router as document_router
from core.config import get_settings
from db.milvus_client import MilvusManager
from services.rag_service import RAGService
from services.vector_service import VectorDocumentService
from utils.bm25_retriever import BM25Retriever
from utils.llm_factory import create_llm_client
from utils.reranker import CrossEncoderReranker

# 全局配置单例，整个进程生命周期内只解析一次
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用启动阶段的资源初始化。

    按顺序完成：
    1. 创建 Milvus 管理器并确保集合存在
    2. 创建文档向量服务（负责 embedding + Milvus 读写）
    3. 创建 LLM 对话客户端（Ollama 或 OpenAI 兼容）
    4. 可选：创建 BM25 召回器（仅当 ENABLE_HYBRID_RECALL=true 时）
    5. 创建 RAG 服务，注入所有依赖

    所有服务实例挂载到 app.state，供路由层通过 Depends 获取。
    """
    # 第一步：Milvus 连接和集合初始化
    milvus_manager = MilvusManager(settings)
    milvus_manager.ensure_collection()
    app.state.settings = settings
    app.state.milvus_manager = milvus_manager

    # 第二步：文档向量服务（embedding 模型在这里初始化）
    vector_service = VectorDocumentService(settings, milvus_manager)
    app.state.vector_service = vector_service

    # 第三步：LLM 对话客户端（根据 LLM_PROVIDER 配置选择 Ollama 或 OpenAI）
    llm_client = create_llm_client(settings)

    # 第四步：BM25 召回器（可选，仅混合召回模式需要）
    # 传入 milvus_client 和 collection_name，用于从 Milvus 拉取文档构建 BM25 索引
    bm25_retriever = (
        BM25Retriever(milvus_manager.client, settings.milvus_collection)
        if settings.enable_hybrid_recall
        else None
    )

    # 第五步：Cross-Encoder 精排器（可选）
    # 懒加载设计：模型在首次 rerank() 调用时才下载和加载，不阻塞启动
    reranker = (
        CrossEncoderReranker(settings.reranker_model)
        if settings.enable_reranker
        else None
    )

    # 第六步：RAG 服务，编排召回 + 粗排 + 精排 + 生成
    app.state.rag_service = RAGService(
        settings, milvus_manager, vector_service.embedding, llm_client,
        bm25_retriever, reranker,
    )

    yield


app = FastAPI(
    title=settings.app_name,
    debug=settings.app_debug,
    version="0.1.0",
    description="A standardized FastAPI service template for Milvus vector database usage.",
    lifespan=lifespan,
)

# 注册文档管理路由（/api/v1/documents/*）
app.include_router(document_router)
# 注册 RAG 问答路由（/api/v1/qa/*）
app.include_router(qa_router)


@app.get("/health")
def health() -> dict:
    """
    基础健康检查接口。

    返回应用状态、Milvus 连接信息和集合详情，
    便于调试时快速确认服务连到了哪个 Milvus 实例。
    """
    info = app.state.milvus_manager.describe_collection()
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "milvus_uri": settings.resolved_milvus_uri,
        "collection_name": settings.milvus_collection,
        "collection_info": info,
    }
