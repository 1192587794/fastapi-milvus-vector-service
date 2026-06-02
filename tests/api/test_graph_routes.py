"""
知识图谱 API 路由测试。

测试图谱管理端点：构建、统计、查询、子图、删除。
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from schemas.graph import Entity, Relation


@pytest.fixture
def mock_graph_service():
    """模拟图谱服务。"""
    service = MagicMock()

    # build_graph_from_document 返回值
    service.build_graph_from_document.return_value = (3, 2)

    # get_stats 返回值
    service.get_stats.return_value = {
        "total_entities": 10,
        "total_relations": 8,
        "entity_type_counts": {"Disease": 5, "Drug": 3, "Symptom": 2},
        "relation_type_counts": {"treats": 4, "causes": 4},
        "documents_count": 3,
    }

    # query_graph 返回值
    service.query_graph.return_value = {
        "entities": [
            Entity(
                id="doc1::entity::高血压::Disease",
                name="高血压",
                type="Disease",
                attributes={},
                doc_id="doc1",
            )
        ],
        "relations": [
            Relation(
                source_id="doc1::entity::高血压::Disease",
                target_id="doc1::entity::头痛::Symptom",
                relation_type="causes",
                confidence=0.8,
                doc_id="doc1",
            )
        ],
        "chunk_ids": ["doc1::chunk::0"],
    }

    # get_subgraph 返回值
    service.get_subgraph.return_value = (
        [],  # nodes
        [],  # edges
    )

    # delete_graph_for_doc 返回值
    service.delete_graph_for_doc.return_value = (3, 2)

    return service


@pytest.fixture
def test_app_with_graph(mock_graph_service):
    """注入 mock 图谱服务的测试客户端。"""
    from main import app

    app.state.graph_service = mock_graph_service
    return TestClient(app)


class TestGraphRoutes:
    """图谱路由测试类。"""

    def test_build_graph(self, test_app_with_graph, mock_graph_service):
        """测试构建图谱。"""
        response = test_app_with_graph.post(
            "/api/v1/graph/build",
            json={"doc_id": "doc1", "text": "患者有高血压，服用阿司匹林治疗。"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "doc1"
        assert data["entities_count"] == 3
        assert data["relations_count"] == 2
        mock_graph_service.build_graph_from_document.assert_called_once()

    def test_build_graph_empty_text(self, test_app_with_graph):
        """测试空文本构建图谱。"""
        response = test_app_with_graph.post(
            "/api/v1/graph/build",
            json={"doc_id": "doc1", "text": ""},
        )
        assert response.status_code == 422  # Pydantic 验证失败

    def test_get_stats(self, test_app_with_graph, mock_graph_service):
        """测试获取统计信息。"""
        response = test_app_with_graph.get("/api/v1/graph/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_entities"] == 10
        assert data["total_relations"] == 8
        assert data["entity_type_counts"]["Disease"] == 5

    def test_query_graph(self, test_app_with_graph, mock_graph_service):
        """测试查询图谱。"""
        response = test_app_with_graph.post(
            "/api/v1/graph/query",
            json={"query": "高血压", "max_hops": 2, "top_k": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "高血压"
        assert len(data["entities"]) == 1
        assert data["entities"][0]["name"] == "高血压"
        assert len(data["relations"]) == 1
        # source_chunks 应该是字符串列表
        assert "doc1::chunk::0" in data["source_chunks"]

    def test_query_graph_empty_query(self, test_app_with_graph):
        """测试空查询。"""
        response = test_app_with_graph.post(
            "/api/v1/graph/query",
            json={"query": "", "max_hops": 2, "top_k": 10},
        )
        assert response.status_code == 422

    def test_get_subgraph(self, test_app_with_graph, mock_graph_service):
        """测试获取子图。"""
        response = test_app_with_graph.post(
            "/api/v1/graph/subgraph",
            json={"entity_name": "高血压", "depth": 1},
        )
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data

    def test_get_subgraph_no_center(self, test_app_with_graph, mock_graph_service):
        """测试获取全图（无中心实体）。"""
        response = test_app_with_graph.post(
            "/api/v1/graph/subgraph",
            json={"entity_name": None, "depth": 1},
        )
        assert response.status_code == 200

    def test_delete_graph(self, test_app_with_graph, mock_graph_service):
        """测试删除图谱。"""
        response = test_app_with_graph.delete("/api/v1/graph/doc1")
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "doc1"
        assert data["deleted_entities"] == 3
        assert data["deleted_relations"] == 2
        mock_graph_service.delete_graph_for_doc.assert_called_once_with("doc1")

    def test_service_not_enabled(self):
        """测试图谱服务未启用时的响应。"""
        from main import app

        # 确保 graph_service 不存在
        if hasattr(app.state, "graph_service"):
            delattr(app.state, "graph_service")

        client = TestClient(app)
        response = client.post(
            "/api/v1/graph/build",
            json={"doc_id": "doc1", "text": "测试文本"},
        )
        assert response.status_code == 503
        assert "not enabled" in response.json()["detail"]
