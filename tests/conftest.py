"""
测试公共 fixtures。

提供 mock 版本的 Settings、MilvusClient、Embedding、LLM 客户端，
让测试不需要真实的 Milvus 服务和 Ollama 服务就能运行。
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from core.config import Settings
from utils.demo_embedding import DemoTextEmbedding


@pytest.fixture
def mock_settings() -> Settings:
    """
    测试用的 Settings 实例，不读取 .env 文件。
    直接通过构造函数参数传入，避免环境变量干扰。
    """
    return Settings(
        _env_file=None,  # 不读 .env
        APP_NAME="Test App",
        APP_ENV="test",
        MILVUS_URI="./data/test.db",
        MILVUS_COLLECTION="test_docs",
        MILVUS_VECTOR_DIMENSION=64,
        OLLAMA_BASE_URL="http://localhost:11434",
        OLLAMA_EMBEDDING_MODEL="test-model",
        OLLAMA_CHAT_MODEL="test-chat-model",
        LLM_PROVIDER="ollama",
        CHUNK_SIZE=100,
        CHUNK_OVERLAP=20,
        RAG_TOP_K=3,
        RAG_RECALL_MULTIPLIER=2,
        ENABLE_HYBRID_RECALL=False,
        HYBRID_RECALL_ALPHA=0.5,
        ENABLE_RERANKER=False,
    )


@pytest.fixture
def mock_milvus_client() -> MagicMock:
    """
    模拟 pymilvus.MilvusClient。
    各测试可以按需设置返回值。
    """
    client = MagicMock()
    client.search.return_value = [[]]
    client.query.return_value = []
    client.upsert.return_value = {"upserted_count": 0}
    client.delete.return_value = {}
    return client


@pytest.fixture
def mock_milvus_manager(mock_milvus_client: MagicMock) -> MagicMock:
    """模拟 MilvusManager，包装 mock_milvus_client。"""
    manager = MagicMock()
    manager.client = mock_milvus_client
    return manager


@pytest.fixture
def mock_embedding() -> DemoTextEmbedding:
    """
    使用 DemoTextEmbedding 替代真实的 Ollama embedding。
    确定性输出，不依赖外部服务。
    """
    return DemoTextEmbedding(dimension=64)


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """
    模拟 LLM 对话客户端。
    chat() 返回固定文本，chat_stream() 逐块返回。
    """
    client = MagicMock()
    client.chat.return_value = "这是一个测试回答。"
    client.chat_stream.return_value = iter(["这是", "一个", "测试", "回答。"])
    return client


@pytest.fixture
def test_app(mock_milvus_manager, mock_embedding, mock_llm_client, mock_settings):
    """
    预注入 mock 服务的 FastAPI TestClient。
    跳过 lifespan 中的 Milvus/Ollama 初始化。
    """
    from main import app
    from services.rag_service import RAGService
    from services.vector_service import VectorDocumentService

    # 创建 mock vector_service
    vector_service = VectorDocumentService(mock_settings, mock_milvus_manager)
    # 替换真实的 embedding 为 mock
    vector_service.embedding = mock_embedding

    # 创建 mock rag_service
    rag_service = RAGService(
        mock_settings, mock_milvus_manager, mock_embedding, mock_llm_client
    )

    # 注入到 app.state
    app.state.settings = mock_settings
    app.state.milvus_manager = mock_milvus_manager
    app.state.vector_service = vector_service
    app.state.rag_service = rag_service

    return TestClient(app)
