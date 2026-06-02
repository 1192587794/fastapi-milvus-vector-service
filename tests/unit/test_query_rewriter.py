"""
QueryRewriter 单元测试。

测试四种 Query 改写策略：查询扩展、HyDE、Step-back、关键词提取。
"""

import json
from unittest.mock import MagicMock

import pytest

from utils.query_rewriter import QueryRewriter, RewrittenQuery


@pytest.fixture
def mock_llm_client():
    """模拟 LLM 客户端。"""
    client = MagicMock()
    return client


@pytest.fixture
def rewriter(mock_llm_client):
    """创建 QueryRewriter 实例。"""
    return QueryRewriter(mock_llm_client)


class TestQueryRewriter:
    """QueryRewriter 测试类。"""

    def test_expand_query(self, rewriter, mock_llm_client):
        """测试查询扩展。"""
        mock_llm_client.chat.return_value = json.dumps([
            "高血压的治疗方法有哪些？",
            "高血压患者应该吃什么药？",
            "如何控制高血压？",
        ])

        result = rewriter._expand_query("高血压怎么治疗？")
        assert len(result) == 3
        assert "高血压" in result[0]

    def test_expand_query_invalid_json(self, rewriter, mock_llm_client):
        """测试查询扩展 - 无效 JSON。"""
        mock_llm_client.chat.return_value = "这不是JSON"

        result = rewriter._expand_query("测试问题")
        assert len(result) == 0

    def test_generate_hyde(self, rewriter, mock_llm_client):
        """测试 HyDE 生成。"""
        mock_llm_client.chat.return_value = "向量数据库是一种专门用于存储和检索高维向量的数据库系统..."

        result = rewriter._generate_hyde("什么是向量数据库？")
        assert len(result) > 0
        assert "向量数据库" in result

    def test_stepback(self, rewriter, mock_llm_client):
        """测试 Step-back 改写。"""
        mock_llm_client.chat.return_value = "高血压患者的用药禁忌有哪些？"

        result = rewriter._stepback("高血压患者能吃阿司匹林吗？")
        assert len(result) > 0
        assert "高血压" in result

    def test_extract_keywords(self, rewriter, mock_llm_client):
        """测试关键词提取。"""
        mock_llm_client.chat.return_value = json.dumps(["阿司匹林", "副作用"])

        result = rewriter._extract_keywords("阿司匹林的副作用有哪些？")
        assert len(result) == 2
        assert "阿司匹林" in result
        assert "副作用" in result

    def test_extract_keywords_invalid_json(self, rewriter, mock_llm_client):
        """测试关键词提取 - 无效 JSON。"""
        mock_llm_client.chat.return_value = "这不是JSON"

        result = rewriter._extract_keywords("测试问题")
        assert len(result) == 0

    def test_rewrite_all_strategies(self, rewriter, mock_llm_client):
        """测试全部策略。"""
        # 设置 LLM 返回不同的结果
        mock_llm_client.chat.side_effect = [
            # 查询扩展
            json.dumps(["子问题1", "子问题2", "子问题3"]),
            # HyDE
            "假设性答案...",
            # Step-back
            "更抽象的问题",
            # 关键词提取
            json.dumps(["关键词1", "关键词2"]),
        ]

        result = rewriter.rewrite("测试问题", strategy="all")

        assert isinstance(result, RewrittenQuery)
        assert result.original == "测试问题"
        assert len(result.expanded) == 3
        assert len(result.hyde_answer) > 0
        assert len(result.stepback) > 0
        assert len(result.keywords) == 2

    def test_rewrite_expansion_only(self, rewriter, mock_llm_client):
        """测试仅查询扩展策略。"""
        mock_llm_client.chat.return_value = json.dumps(["子问题1", "子问题2"])

        result = rewriter.rewrite("测试问题", strategy="expansion")

        assert len(result.expanded) == 2
        assert len(result.hyde_answer) == 0
        assert len(result.stepback) == 0
        assert len(result.keywords) == 0

    def test_rewrite_hyde_only(self, rewriter, mock_llm_client):
        """测试仅 HyDE 策略。"""
        mock_llm_client.chat.return_value = "假设性答案"

        result = rewriter.rewrite("测试问题", strategy="hyde")

        assert len(result.expanded) == 0
        assert len(result.hyde_answer) > 0
        assert len(result.stepback) == 0
        assert len(result.keywords) == 0

    def test_rewrite_stepback_only(self, rewriter, mock_llm_client):
        """测试仅 Step-back 策略。"""
        mock_llm_client.chat.return_value = "抽象问题"

        result = rewriter.rewrite("测试问题", strategy="stepback")

        assert len(result.expanded) == 0
        assert len(result.hyde_answer) == 0
        assert len(result.stepback) > 0
        assert len(result.keywords) == 0

    def test_rewrite_keywords_only(self, rewriter, mock_llm_client):
        """测试仅关键词提取策略。"""
        mock_llm_client.chat.return_value = json.dumps(["关键词1", "关键词2"])

        result = rewriter.rewrite("测试问题", strategy="keywords")

        assert len(result.expanded) == 0
        assert len(result.hyde_answer) == 0
        assert len(result.stepback) == 0
        assert len(result.keywords) == 2

    def test_rewrite_llm_exception(self, rewriter, mock_llm_client):
        """测试 LLM 调用异常。"""
        mock_llm_client.chat.side_effect = Exception("LLM 服务不可用")

        result = rewriter.rewrite("测试问题", strategy="all")

        # 异常时应该返回空结果，而不是抛出异常
        assert isinstance(result, RewrittenQuery)
        assert result.original == "测试问题"
        assert len(result.expanded) == 0
        assert len(result.hyde_answer) == 0
        assert len(result.stepback) == 0
        assert len(result.keywords) == 0

    def test_parse_json_list_normal(self, rewriter):
        """测试解析正常 JSON 数组。"""
        response = '["item1", "item2", "item3"]'
        result = rewriter._parse_json_list(response)
        assert len(result) == 3
        assert result[0] == "item1"

    def test_parse_json_list_with_extra_text(self, rewriter):
        """测试解析带额外文本的 JSON 数组。"""
        response = '这是结果：["item1", "item2"] 以上。'
        result = rewriter._parse_json_list(response)
        assert len(result) == 2

    def test_parse_json_list_empty(self, rewriter):
        """测试解析空响应。"""
        response = ''
        result = rewriter._parse_json_list(response)
        assert len(result) == 0

    def test_parse_json_list_no_array(self, rewriter):
        """测试解析没有数组的响应。"""
        response = '{"error": "invalid"}'
        result = rewriter._parse_json_list(response)
        assert len(result) == 0

    def test_parse_json_list_invalid_json(self, rewriter):
        """测试解析无效 JSON。"""
        response = '[invalid json'
        result = rewriter._parse_json_list(response)
        assert len(result) == 0


class TestRewrittenQuery:
    """RewrittenQuery 数据结构测试。"""

    def test_default_values(self):
        """测试默认值。"""
        query = RewrittenQuery()
        assert query.original == ""
        assert query.expanded == []
        assert query.hyde_answer == ""
        assert query.stepback == ""
        assert query.keywords == []

    def test_custom_values(self):
        """测试自定义值。"""
        query = RewrittenQuery(
            original="测试问题",
            expanded=["子问题1", "子问题2"],
            hyde_answer="假设性答案",
            stepback="抽象问题",
            keywords=["关键词1", "关键词2"],
        )
        assert query.original == "测试问题"
        assert len(query.expanded) == 2
        assert query.hyde_answer == "假设性答案"
        assert query.stepback == "抽象问题"
        assert len(query.keywords) == 2
