"""
EntityExtractor 单元测试。

测试 LLM 实体抽取的各种场景：正常抽取、JSON 解析失败、空文本。
"""

import json
from unittest.mock import MagicMock

import pytest

from utils.entity_extractor import EntityExtractor


@pytest.fixture
def mock_llm_client():
    """模拟 LLM 客户端。"""
    client = MagicMock()
    return client


@pytest.fixture
def extractor(mock_llm_client):
    """创建实体抽取器实例。"""
    return EntityExtractor(mock_llm_client)


class TestEntityExtractor:
    """EntityExtractor 测试类。"""

    def test_extract_normal(self, extractor, mock_llm_client):
        """测试正常实体抽取。"""
        mock_llm_client.chat.return_value = json.dumps([
            {"name": "高血压", "type": "Disease", "attributes": {"描述": "血压升高"}},
            {"name": "阿司匹林", "type": "Drug", "attributes": {"描述": "抗血小板药"}},
        ])

        entities = extractor.extract("患者有高血压，服用阿司匹林。", doc_id="doc1")
        assert len(entities) == 2
        assert entities[0].name == "高血压"
        assert entities[0].type == "Disease"
        assert entities[1].name == "阿司匹林"
        assert entities[1].type == "Drug"
        assert entities[0].doc_id == "doc1"

    def test_extract_with_chunk_id(self, extractor, mock_llm_client):
        """测试带 chunk_id 的抽取。"""
        mock_llm_client.chat.return_value = json.dumps([
            {"name": "头痛", "type": "Symptom", "attributes": {}}
        ])

        entities = extractor.extract("患者头痛。", doc_id="doc1", chunk_id="doc1::chunk::0")
        assert len(entities) == 1
        assert entities[0].chunk_id == "doc1::chunk::0"

    def test_extract_empty_text(self, extractor):
        """测试空文本。"""
        entities = extractor.extract("", doc_id="doc1")
        assert len(entities) == 0

    def test_extract_whitespace_text(self, extractor):
        """测试纯空白文本。"""
        entities = extractor.extract("   ", doc_id="doc1")
        assert len(entities) == 0

    def test_extract_invalid_json(self, extractor, mock_llm_client):
        """测试 LLM 返回无效 JSON。"""
        mock_llm_client.chat.return_value = "这不是JSON"

        entities = extractor.extract("测试文本", doc_id="doc1")
        assert len(entities) == 0

    def test_extract_partial_json(self, extractor, mock_llm_client):
        """测试 LLM 返回带额外文本的 JSON。"""
        mock_llm_client.chat.return_value = '这是抽取结果：[{"name": "高血压", "type": "Disease", "attributes": {}}] 以上。'

        entities = extractor.extract("测试文本", doc_id="doc1")
        assert len(entities) == 1
        assert entities[0].name == "高血压"

    def test_extract_invalid_entity_type(self, extractor, mock_llm_client):
        """测试无效的实体类型。"""
        mock_llm_client.chat.return_value = json.dumps([
            {"name": "某实体", "type": "InvalidType", "attributes": {}}
        ])

        entities = extractor.extract("测试文本", doc_id="doc1")
        assert len(entities) == 1
        assert entities[0].type == "Other"  # 无效类型应归为 Other

    def test_extract_empty_name(self, extractor, mock_llm_client):
        """测试空名称的实体。"""
        mock_llm_client.chat.return_value = json.dumps([
            {"name": "", "type": "Disease", "attributes": {}},
            {"name": "高血压", "type": "Disease", "attributes": {}},
        ])

        entities = extractor.extract("测试文本", doc_id="doc1")
        assert len(entities) == 1  # 空名称应被过滤
        assert entities[0].name == "高血压"

    def test_extract_dedup(self, extractor, mock_llm_client):
        """测试实体去重。"""
        mock_llm_client.chat.return_value = json.dumps([
            {"name": "高血压", "type": "Disease", "attributes": {}},
            {"name": "高血压", "type": "Disease", "attributes": {"描述": "重复"}},
        ])

        entities = extractor.extract("测试文本", doc_id="doc1")
        assert len(entities) == 1  # 重复实体应被去重

    def test_extract_llm_exception(self, extractor, mock_llm_client):
        """测试 LLM 调用异常。"""
        mock_llm_client.chat.side_effect = Exception("LLM 服务不可用")

        entities = extractor.extract("测试文本", doc_id="doc1")
        assert len(entities) == 0  # 异常应返回空列表

    def test_extract_non_list_response(self, extractor, mock_llm_client):
        """测试 LLM 返回非数组 JSON。"""
        mock_llm_client.chat.return_value = '{"error": "invalid"}'

        entities = extractor.extract("测试文本", doc_id="doc1")
        assert len(entities) == 0

    def test_extract_non_dict_items(self, extractor, mock_llm_client):
        """测试 LLM 返回数组中包含非字典元素。"""
        mock_llm_client.chat.return_value = '["not", "dict", 123]'

        entities = extractor.extract("测试文本", doc_id="doc1")
        assert len(entities) == 0
