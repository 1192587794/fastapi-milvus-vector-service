"""LLM 工厂函数的单元测试。"""

from unittest.mock import MagicMock

import pytest

from utils.llm_factory import create_llm_client


class TestCreateLLMClient:
    """测试 LLM 客户端工厂函数。"""

    def test_ollama_provider(self, mock_settings):
        """provider=ollama 返回 OllamaChatClient。"""
        from utils.ollama_chat import OllamaChatClient

        client = create_llm_client(mock_settings)
        assert isinstance(client, OllamaChatClient)

    def test_openai_provider(self, mock_settings):
        """provider=openai 返回 OpenAIChatClient。"""
        from utils.openai_chat import OpenAIChatClient

        mock_settings.llm_provider = "openai"
        mock_settings.openai_api_key = "test-key"
        client = create_llm_client(mock_settings)
        assert isinstance(client, OpenAIChatClient)

    def test_openai_no_api_key_raises(self, mock_settings):
        """provider=openai 但没有 api_key 时抛出 ValueError。"""
        mock_settings.llm_provider = "openai"
        mock_settings.openai_api_key = None
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_llm_client(mock_settings)

    def test_unsupported_provider_raises(self, mock_settings):
        """不支持的 provider 抛出 ValueError。"""
        mock_settings.llm_provider = "anthropic"
        with pytest.raises(ValueError, match="不支持"):
            create_llm_client(mock_settings)

    def test_case_insensitive(self, mock_settings):
        """provider 名称大小写不敏感。"""
        from utils.ollama_chat import OllamaChatClient

        mock_settings.llm_provider = "Ollama"
        client = create_llm_client(mock_settings)
        assert isinstance(client, OllamaChatClient)
