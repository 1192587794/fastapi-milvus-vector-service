from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    统一的应用配置类。

    这样做的目的，是把所有运行参数都集中管理，避免在代码里到处散落硬编码。
    当项目从本地开发切换到测试环境、生产环境时，只需要改环境变量，不需要改代码。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="Milvus FastAPI", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")

    # 这里同时支持两种连接方式：
    # 1. 文件路径：走 Milvus Lite，适合本地开发和快速演示。
    # 2. HTTP 地址：连接远程 Milvus / Zilliz Cloud，适合真实部署。
    milvus_uri: str = Field(default="./data/milvus_demo.db", alias="MILVUS_URI")
    milvus_token: str | None = Field(default=None, alias="MILVUS_TOKEN")
    milvus_db_name: str = Field(default="default", alias="MILVUS_DB_NAME")

    milvus_collection: str = Field(default="documents", alias="MILVUS_COLLECTION")
    milvus_vector_dimension: int = Field(default=64, alias="MILVUS_VECTOR_DIMENSION")
    milvus_metric_type: str = Field(default="COSINE", alias="MILVUS_METRIC_TYPE")
    milvus_consistency_level: str = Field(default="Bounded", alias="MILVUS_CONSISTENCY_LEVEL")
    milvus_drop_existing_on_start: bool = Field(default=False, alias="MILVUS_DROP_EXISTING_ON_START")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")

    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")

    upload_max_file_size_mb: int = Field(default=50, alias="UPLOAD_MAX_FILE_SIZE_MB")
    upload_allowed_extensions: list[str] = Field(
        default=[".pdf", ".docx"], alias="UPLOAD_ALLOWED_EXTENSIONS"
    )

    # --- LLM / RAG 配置 ---
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")  # ollama 或 openai
    ollama_chat_model: str = Field(default="qwen2.5:7b", alias="OLLAMA_CHAT_MODEL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    rag_recall_multiplier: int = Field(default=4, alias="RAG_RECALL_MULTIPLIER")

    @property
    def resolved_milvus_uri(self) -> str:
        """
        把相对路径形式的本地数据库地址转换成绝对路径。

        这样做能避免服务从不同工作目录启动时，Milvus Lite 的数据库文件被创建到意料之外的位置。
        如果配置本身就是 http/https 地址，则原样返回。
        """
        if self.milvus_uri.startswith(("http://", "https://")):
            return self.milvus_uri

        return str((Path.cwd() / self.milvus_uri).resolve())


@lru_cache
def get_settings() -> Settings:
    """使用缓存避免在一次进程生命周期内重复解析配置文件。"""
    return Settings()
