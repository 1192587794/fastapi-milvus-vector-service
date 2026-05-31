"""BM25 稀疏召回模块的单元测试。"""

from unittest.mock import MagicMock

import pytest

from utils.bm25_retriever import BM25Retriever, _tokenize


class TestTokenize:
    """测试 jieba 分词函数。"""

    def test_chinese_text(self):
        """中文文本分词。"""
        tokens = _tokenize("什么是向量数据库")
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_english_text(self):
        """英文文本分词。"""
        tokens = _tokenize("what is vector database")
        assert "vector" in [t.lower() for t in tokens]

    def test_mixed_text(self):
        """中英文混合文本分词。"""
        tokens = _tokenize("Milvus 是一个向量数据库")
        assert len(tokens) > 0

    def test_stopwords_filtered(self):
        """停用词应该被过滤。"""
        tokens = _tokenize("这是一个测试")
        # "这"、"是"、"一个" 都是停用词
        assert "这" not in tokens
        assert "是" not in tokens

    def test_punctuation_filtered(self):
        """纯标点应该被过滤。"""
        tokens = _tokenize("，。！？")
        assert len(tokens) == 0

    def test_empty_string(self):
        """空字符串返回空列表。"""
        tokens = _tokenize("")
        assert tokens == []


class TestBM25Retriever:
    """测试 BM25 召回器。"""

    def test_retrieve_returns_results(self, mock_milvus_client):
        """正常召回返回结果。"""
        mock_milvus_client.query.return_value = [
            {"id": "doc1", "text": "向量数据库是一种数据库", "source": "test", "metadata": {}},
            {"id": "doc2", "text": "Milvus 是流行的向量数据库", "source": "test", "metadata": {}},
            {"id": "doc3", "text": "今天天气很好", "source": "test", "metadata": {}},
        ]

        retriever = BM25Retriever(mock_milvus_client, "test_collection")
        results = retriever.retrieve(query="向量数据库", top_k=2)

        assert len(results) == 2
        assert all("bm25_score" in r for r in results)
        assert all("id" in r for r in results)
        assert all("text" in r for r in results)

    def test_retrieve_empty_candidates(self, mock_milvus_client):
        """Milvus 返回空候选时，返回空列表。"""
        mock_milvus_client.query.return_value = []
        retriever = BM25Retriever(mock_milvus_client, "test_collection")
        results = retriever.retrieve(query="测试", top_k=5)
        assert results == []

    def test_retrieve_milvus_error_returns_empty(self, mock_milvus_client):
        """Milvus 查询异常时，优雅降级返回空列表。"""
        mock_milvus_client.query.side_effect = Exception("连接失败")
        retriever = BM25Retriever(mock_milvus_client, "test_collection")
        results = retriever.retrieve(query="测试", top_k=5)
        assert results == []

    def test_retrieve_top_k_limit(self, mock_milvus_client):
        """返回结果数量不超过 top_k。"""
        mock_milvus_client.query.return_value = [
            {"id": f"doc{i}", "text": f"文档{i} 内容", "source": "test", "metadata": {}}
            for i in range(10)
        ]
        retriever = BM25Retriever(mock_milvus_client, "test_collection")
        results = retriever.retrieve(query="文档", top_k=3)
        assert len(results) <= 3

    def test_retrieve_result_structure(self, mock_milvus_client):
        """返回结果包含所有必要字段。"""
        mock_milvus_client.query.return_value = [
            {"id": "doc1", "text": "测试文本", "source": "upload", "metadata": {"key": "value"}},
        ]
        retriever = BM25Retriever(mock_milvus_client, "test_collection")
        results = retriever.retrieve(query="测试", top_k=5)

        assert len(results) == 1
        r = results[0]
        assert r["id"] == "doc1"
        assert r["text"] == "测试文本"
        assert r["source"] == "upload"
        assert r["metadata"] == {"key": "value"}
        assert isinstance(r["bm25_score"], float)

    def test_fetch_candidates_passes_filter(self, mock_milvus_client):
        """filter_expr 应该传递给 Milvus query。"""
        mock_milvus_client.query.return_value = []
        retriever = BM25Retriever(mock_milvus_client, "test_collection")
        retriever.retrieve(query="测试", top_k=5, filter_expr='source == "upload"')

        mock_milvus_client.query.assert_called_once()
        call_kwargs = mock_milvus_client.query.call_args
        assert call_kwargs[1]["filter"] == 'source == "upload"'
