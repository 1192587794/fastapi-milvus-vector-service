"""文档管理 API 端点测试。"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class TestUpsertEndpoint:
    """测试 POST /api/v1/documents/upsert。"""

    def test_upsert_success(self, test_app: TestClient, mock_milvus_client):
        """正常插入文档。"""
        mock_milvus_client.upsert.return_value = {"upserted_count": 1}
        response = test_app.post(
            "/api/v1/documents/upsert",
            json={"items": [{"id": "doc1", "text": "测试文档内容"}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["upserted_count"] >= 0

    def test_upsert_empty_items_validation(self, test_app: TestClient):
        """空 items 列表应该校验失败。"""
        response = test_app.post(
            "/api/v1/documents/upsert",
            json={"items": []},
        )
        assert response.status_code == 422

    def test_upsert_empty_text_validation(self, test_app: TestClient):
        """text 为空应该校验失败。"""
        response = test_app.post(
            "/api/v1/documents/upsert",
            json={"items": [{"id": "doc1", "text": ""}]},
        )
        assert response.status_code == 422


class TestSearchEndpoint:
    """测试 POST /api/v1/documents/search。"""

    def test_search_success(self, test_app: TestClient, mock_milvus_client):
        """正常搜索。"""
        mock_milvus_client.search.return_value = [[]]
        response = test_app.post(
            "/api/v1/documents/search",
            json={"query_text": "测试查询"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "hits" in data
        assert "query_text" in data

    def test_search_empty_query_validation(self, test_app: TestClient):
        """空查询应该校验失败。"""
        response = test_app.post(
            "/api/v1/documents/search",
            json={"query_text": ""},
        )
        assert response.status_code == 422

    def test_search_with_source_filter(self, test_app: TestClient, mock_milvus_client):
        """带 source 过滤的搜索。"""
        mock_milvus_client.search.return_value = [[]]
        response = test_app.post(
            "/api/v1/documents/search",
            json={"query_text": "测试", "source": "upload"},
        )
        assert response.status_code == 200


class TestGetDocumentEndpoint:
    """测试 GET /api/v1/documents/{id}。"""

    def test_get_not_found(self, test_app: TestClient, mock_milvus_client):
        """查询不存在的文档返回 404。"""
        mock_milvus_client.query.return_value = []
        response = test_app.get("/api/v1/documents/nonexistent")
        assert response.status_code == 404


class TestDeleteEndpoint:
    """测试 DELETE /api/v1/documents/{id}。"""

    def test_delete_nonexistent(self, test_app: TestClient, mock_milvus_client):
        """删除不存在的文档。"""
        mock_milvus_client.query.return_value = []
        response = test_app.delete("/api/v1/documents/nonexistent")
        # 可能返回 404 或 200（取决于实现）
        assert response.status_code in (200, 404)


class TestHealthEndpoint:
    """测试 GET /health。"""

    def test_health(self, test_app: TestClient, mock_milvus_manager):
        """健康检查端点。"""
        mock_milvus_manager.describe_collection.return_value = {"count": 0}
        response = test_app.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
