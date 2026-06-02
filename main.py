"""
FastAPI 应用入口。

负责：
1. 读取全局配置
2. 在 lifespan 中初始化所有服务组件
3. 注册路由
4. 定义健康检查端点
"""

import logging
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.qa_routes import router as qa_router
from api.routes import router as document_router
from core.config import get_settings
from db.milvus_client import MilvusManager
from services.rag_service import RAGService
from services.session_service import SessionService
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

    # 第二步：LLM 对话客户端（根据 LLM_PROVIDER 配置选择 Ollama 或 OpenAI）
    llm_client = create_llm_client(settings)

    # 第三步：知识图谱服务（可选，需要在 vector_service 之前初始化）
    # 启用时，文档入库自动构建图谱，RAG 检索增加图谱召回路
    graph_service = None
    graph_retriever = None
    if settings.enable_knowledge_graph:
        from db.graph_store import create_graph_store
        from services.graph_service import GraphService
        from utils.entity_extractor import EntityExtractor
        from utils.relation_extractor import RelationExtractor
        from utils.graph_retriever import GraphRetriever

        graph_store = create_graph_store(settings)
        entity_extractor = EntityExtractor(llm_client, settings)
        relation_extractor = RelationExtractor(llm_client, settings)
        # embedding 模型在 vector_service 中创建，这里先传 None
        graph_service = GraphService(
            settings, graph_store, entity_extractor,
            relation_extractor, milvus_manager, None,
        )
        graph_retriever = GraphRetriever(graph_service, milvus_manager, settings)
        app.state.graph_service = graph_service
        logging.getLogger(__name__).info("知识图谱服务已启用 (backend=%s)", settings.graph_store_backend)

    # 第四步：文档向量服务（embedding 模型在这里初始化）
    # 传入 graph_service，文档入库时自动触发图谱构建
    vector_service = VectorDocumentService(settings, milvus_manager, graph_service)
    app.state.vector_service = vector_service

    # 如果图谱服务已创建，更新其 embedding 模型引用
    if graph_service:
        graph_service._embedding = vector_service.embedding

    # 第五步：BM25 召回器（可选，仅混合召回模式需要）
    # 传入 milvus_client 和 collection_name，用于从 Milvus 拉取文档构建 BM25 索引
    bm25_retriever = (
        BM25Retriever(milvus_manager.client, settings.milvus_collection)
        if settings.enable_hybrid_recall
        else None
    )

    # 第六步：Cross-Encoder 精排器（可选）
    # 懒加载设计：模型在首次 rerank() 调用时才下载和加载，不阻塞启动
    reranker = (
        CrossEncoderReranker(settings.reranker_model)
        if settings.enable_reranker
        else None
    )

    # 第七步：Redis 会话服务（可选）
    # 用于存储多轮对话历史，客户端只需传 session_id 即可续接对话
    session_service = None
    try:
        redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        redis_client.ping()  # 测试 Redis 连接
        session_service = SessionService(redis_client, settings.session_ttl_seconds)
        logging.getLogger(__name__).info("Redis 会话服务已连接: %s", settings.redis_url)
    except redis.RedisError as e:
        logging.getLogger(__name__).warning(
            "Redis 连接失败，多轮对话服务端存储不可用: %s。客户端仍可通过 history 字段传入历史。",
            e,
        )

    # 第八步：Query 改写器（可选）
    # 启用时，在召回前对问题进行改写，提高召回率和精度
    query_rewriter = None
    if settings.enable_query_rewrite:
        from utils.query_rewriter import QueryRewriter
        query_rewriter = QueryRewriter(llm_client, settings)
        logging.getLogger(__name__).info("Query 改写已启用 (strategy=%s)", settings.query_rewrite_strategy)

    # 第九步：RAG 服务，编排召回 + 粗排 + 精排 + 生成
    app.state.rag_service = RAGService(
        settings, milvus_manager, vector_service.embedding, llm_client,
        bm25_retriever, reranker, session_service, graph_retriever, query_rewriter,
    )

    yield


app = FastAPI(
    title=settings.app_name,
    debug=settings.app_debug,
    version="0.1.0",
    description="A standardized FastAPI service template for Milvus vector database usage.",
    lifespan=lifespan,
)

# CORS 跨域配置（前端开发服务器需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册文档管理路由（/api/v1/documents/*）
app.include_router(document_router)
# 注册 RAG 问答路由（/api/v1/qa/*）
app.include_router(qa_router)
# 注册知识图谱路由（/api/v1/graph/*）
from api.graph_routes import router as graph_router
app.include_router(graph_router)


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
