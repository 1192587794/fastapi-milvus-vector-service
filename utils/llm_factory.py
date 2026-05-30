"""LLM 客户端工厂，根据配置返回 Ollama 或 OpenAI 兼容客户端。"""

from core.config import Settings
from utils.ollama_chat import OllamaChatClient
from utils.openai_chat import OpenAIChatClient


def create_llm_client(settings: Settings) -> OllamaChatClient | OpenAIChatClient:
    """根据 settings.llm_provider 创建对应的对话客户端。"""
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
