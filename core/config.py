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

    # --- 基础应用配置 ---
    app_name: str = Field(default="Milvus FastAPI", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")

    # --- Milvus 数据库配置 ---
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

    # --- Embedding 模型配置 ---
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")

    # --- 文本分片配置 ---
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")

    # --- 文件上传配置 ---
    upload_max_file_size_mb: int = Field(default=50, alias="UPLOAD_MAX_FILE_SIZE_MB")
    upload_allowed_extensions: list[str] = Field(
        default=[".pdf", ".docx"], alias="UPLOAD_ALLOWED_EXTENSIONS"
    )

    # --- LLM / RAG 配置 ---
    # llm_provider 决定使用哪个 LLM 后端：
    #   "ollama" -> 使用本地 Ollama 服务，需要配置 OLLAMA_CHAT_MODEL
    #   "openai" -> 使用 OpenAI 兼容 API（支持 DeepSeek、硅基流动等），需要配置 OPENAI_API_KEY
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    ollama_chat_model: str = Field(default="qwen2.5:7b", alias="OLLAMA_CHAT_MODEL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")

    # rag_top_k: 最终送入 LLM 的上下文文档数量
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    # rag_recall_multiplier: 召回时的倍数，实际召回 top_k * multiplier 条候选
    # 例如 top_k=5, multiplier=4 -> 召回 20 条候选，经排序后取前 5 条
    rag_recall_multiplier: int = Field(default=4, alias="RAG_RECALL_MULTIPLIER")

    # --- 混合召回配置 ---
    # enable_hybrid_recall: 是否开启混合召回（向量 + BM25）
    #   false（默认）-> 仅向量召回
    #   true -> 向量召回 + BM25 召回 + RRF 粗排融合
    enable_hybrid_recall: bool = Field(default=False, alias="ENABLE_HYBRID_RECALL")
    # hybrid_recall_alpha: RRF 融合权重（0-1）
    #   0.5（默认）-> 向量和 BM25 等权重
    #   越大 -> 越偏向向量召回（语义匹配）
    #   越小 -> 越偏向 BM25 召回（关键词匹配）
    hybrid_recall_alpha: float = Field(default=0.5, alias="HYBRID_RECALL_ALPHA")

    # --- 精排配置 ---
    # enable_reranker: 是否开启 Cross-Encoder 精排
    #   false（默认）-> 不精排，直接用粗排结果
    #   true -> 在粗排后用 Cross-Encoder 模型对候选文档逐对打分
    enable_reranker: bool = Field(default=False, alias="ENABLE_RERANKER")
    # reranker_model: Hugging Face 上的 Cross-Encoder 模型名称
    #   首次使用时自动下载（约 80MB），后续使用本地缓存
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )

    # --- 多轮对话配置 ---
    # rag_max_history_turns: 最多保留多少轮对话历史
    #   10（默认）-> 保留最近 10 轮（user + assistant 各 10 条）
    #   设为 0 则禁用对话历史，每次都像全新对话
    #   过长的历史会占用 LLM 的上下文窗口，影响参考资料的可用空间
    rag_max_history_turns: int = Field(default=10, alias="RAG_MAX_HISTORY_TURNS")

    # --- Redis 会话存储配置 ---
    # redis_url: Redis 连接地址，用于存储多轮对话历史
    #   默认连接本地 Redis 的 0 号数据库
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    # session_ttl_seconds: 会话过期时间（秒），过期后自动清理
    #   3600（默认）-> 1 小时无活动后过期
    session_ttl_seconds: int = Field(default=3600, alias="SESSION_TTL_SECONDS")

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
