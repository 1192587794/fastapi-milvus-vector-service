"""
LLM 客户端工厂

根据配置中的 LLM_PROVIDER 字段，创建对应的对话客户端实例。
这样 RAG 服务层不需要关心具体用的是 Ollama 还是 OpenAI，只需调用统一的 chat/chat_stream 接口。
"""

from core.config import Settings
from utils.ollama_chat import OllamaChatClient
from utils.openai_chat import OpenAIChatClient


def create_llm_client(settings: Settings) -> OllamaChatClient | OpenAIChatClient:
    """
    根据 settings.llm_provider 创建对应的对话客户端。

    支持的 provider 值：
    - "ollama": 使用本地 Ollama 服务，需要 OLLAMA_BASE_URL 和 OLLAMA_CHAT_MODEL
    - "openai": 使用 OpenAI 兼容 API，需要 OPENAI_API_KEY、OPENAI_BASE_URL 和 OPENAI_CHAT_MODEL

    参数:
        settings: 应用配置对象

    返回:
        OllamaChatClient 或 OpenAIChatClient 实例

    异常:
        ValueError: 当 provider 值不支持，或 openai 模式下缺少 API key 时
    """
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return OllamaChatClient(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
        )
    elif provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("LLM_PROVIDER 设置为 openai 时，OPENAI_API_KEY 不能为空。")
        return OpenAIChatClient(
            model=settings.openai_chat_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
    else:
        raise ValueError(f"不支持的 LLM_PROVIDER: {provider}，可选值: ollama, openai")
