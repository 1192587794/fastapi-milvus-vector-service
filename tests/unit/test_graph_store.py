"""
NetworkXGraphStore 单元测试。

测试图存储的核心功能：增删改查、多跳遍历、持久化。
"""

import json
import tempfile
from pathlib import Path

import pytest

from db.graph_store import NetworkXGraphStore
from schemas.graph import Entity, Relation


@pytest.fixture
def graph_store():
    """创建一个不持久化的内存图存储。"""
    return NetworkXGraphStore()


@pytest.fixture
def sample_entities():
    """示例实体列表。"""
    return [
        Entity(
            id="doc1::entity::高血压::Disease",
            name="高血压",
            type="Disease",
            attributes={"描述": "血压持续升高"},
            doc_id="doc1",
            chunk_id="doc1::chunk::0",
        ),
        Entity(
            id="doc1::entity::阿司匹林::Drug",
            name="阿司匹林",
            type="Drug",
            attributes={"描述": "抗血小板药物"},
            doc_id="doc1",
            chunk_id="doc1::chunk::1",
        ),
        Entity(
            id="doc1::entity::头痛::Symptom",
            name="头痛",
            type="Symptom",
            attributes={"描述": "头部疼痛"},
            doc_id="doc1",
            chunk_id="doc1::chunk::0",
        ),
    ]


@pytest.fixture
def sample_relations():
    """示例关系列表。"""
    return [
        Relation(
            source_id="doc1::entity::阿司匹林::Drug",
            target_id="doc1::entity::头痛::Symptom",
            relation_type="treats",
            confidence=0.9,
            doc_id="doc1",
            chunk_id="doc1::chunk::1",
        ),
        Relation(
            source_id="doc1::entity::高血压::Disease",
            target_id="doc1::entity::头痛::Symptom",
            relation_type="causes",
            confidence=0.8,
            doc_id="doc1",
            chunk_id="doc1::chunk::0",
        ),
    ]


class TestNetworkXGraphStore:
    """NetworkXGraphStore 测试类。"""

    def test_add_entities(self, graph_store, sample_entities):
        """测试添加实体。"""
        added = graph_store.add_entities(sample_entities)
        assert added == 3
        assert graph_store._graph.number_of_nodes() == 3

    def test_add_entities_dedup(self, graph_store, sample_entities):
        """测试重复添加实体不会增加节点数。"""
        graph_store.add_entities(sample_entities)
        added = graph_store.add_entities(sample_entities[:1])
        assert added == 0  # 已存在，不增加
        assert graph_store._graph.number_of_nodes() == 3

    def test_add_relations(self, graph_store, sample_entities, sample_relations):
        """测试添加关系。"""
        graph_store.add_entities(sample_entities)
        added = graph_store.add_relations(sample_relations)
        assert added == 2
        assert graph_store._graph.number_of_edges() == 2

    def test_query_entity_exact(self, graph_store, sample_entities):
        """测试精确查询实体。"""
        graph_store.add_entities(sample_entities)
        results = graph_store.query_entity("高血压", fuzzy=False)
        assert len(results) == 1
        assert results[0].name == "高血压"

    def test_query_entity_fuzzy(self, graph_store, sample_entities):
        """测试模糊查询实体。"""
        graph_store.add_entities(sample_entities)
        results = graph_store.query_entity("血压", fuzzy=True)
        assert len(results) == 1
        assert results[0].name == "高血压"

    def test_query_entity_not_found(self, graph_store, sample_entities):
        """测试查询不存在的实体。"""
        graph_store.add_entities(sample_entities)
        results = graph_store.query_entity("糖尿病", fuzzy=False)
        assert len(results) == 0

    def test_query_neighbors_1hop(self, graph_store, sample_entities, sample_relations):
        """测试单跳邻居查询。"""
        graph_store.add_entities(sample_entities)
        graph_store.add_relations(sample_relations)

        entities, relations = graph_store.query_neighbors(
            "doc1::entity::阿司匹林::Drug", max_hops=1
        )
        # 阿司匹林 -> 头痛（出边）
        assert len(entities) >= 2  # 至少包含阿司匹林和头痛
        assert len(relations) >= 1

    def test_query_neighbors_2hop(self, graph_store, sample_entities, sample_relations):
        """测试两跳邻居查询。"""
        graph_store.add_entities(sample_entities)
        graph_store.add_relations(sample_relations)

        entities, relations = graph_store.query_neighbors(
            "doc1::entity::阿司匹林::Drug", max_hops=2
        )
        # 阿司匹林 -> 头痛 <- 高血压（两跳）
        assert len(entities) == 3  # 三个实体都应该可达
        assert len(relations) == 2

    def test_query_neighbors_not_found(self, graph_store):
        """测试查询不存在实体的邻居。"""
        entities, relations = graph_store.query_neighbors("nonexistent", max_hops=1)
        assert len(entities) == 0
        assert len(relations) == 0

    def test_delete_by_doc(self, graph_store, sample_entities, sample_relations):
        """测试按文档 ID 删除。"""
        graph_store.add_entities(sample_entities)
        graph_store.add_relations(sample_relations)

        deleted_entities, deleted_relations = graph_store.delete_by_doc("doc1")
        assert deleted_entities == 3
        assert deleted_relations == 2
        assert graph_store._graph.number_of_nodes() == 0
        assert graph_store._graph.number_of_edges() == 0

    def test_get_stats(self, graph_store, sample_entities, sample_relations):
        """测试获取统计信息。"""
        graph_store.add_entities(sample_entities)
        graph_store.add_relations(sample_relations)

        stats = graph_store.get_stats()
        assert stats["total_entities"] == 3
        assert stats["total_relations"] == 2
        assert stats["entity_type_counts"]["Disease"] == 1
        assert stats["entity_type_counts"]["Drug"] == 1
        assert stats["entity_type_counts"]["Symptom"] == 1
        assert stats["relation_type_counts"]["treats"] == 1
        assert stats["relation_type_counts"]["causes"] == 1
        assert stats["documents_count"] == 1

    def test_get_subgraph_with_center(self, graph_store, sample_entities, sample_relations):
        """测试以指定实体为中心获取子图。"""
        graph_store.add_entities(sample_entities)
        graph_store.add_relations(sample_relations)

        nodes, edges = graph_store.get_subgraph(center_name="阿司匹林", depth=1)
        assert len(nodes) >= 2
        assert len(edges) >= 1

    def test_get_subgraph_without_center(self, graph_store, sample_entities, sample_relations):
        """测试获取全图摘要。"""
        graph_store.add_entities(sample_entities)
        graph_store.add_relations(sample_relations)

        nodes, edges = graph_store.get_subgraph(center_name=None, depth=1)
        assert len(nodes) == 3
        assert len(edges) == 2

    def test_save_and_load(self, sample_entities, sample_relations):
        """测试持久化和加载。"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            persist_path = f.name

        try:
            # 创建并保存
            store1 = NetworkXGraphStore(persist_path=persist_path)
            store1.add_entities(sample_entities)
            store1.add_relations(sample_relations)
            store1.save()

            # 验证文件存在
            assert Path(persist_path).exists()

            # 加载到新实例
            store2 = NetworkXGraphStore(persist_path=persist_path)
            assert store2._graph.number_of_nodes() == 3
            assert store2._graph.number_of_edges() == 2

            # 验证数据正确性
            stats = store2.get_stats()
            assert stats["total_entities"] == 3
            assert stats["total_relations"] == 2
        finally:
            Path(persist_path).unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件。"""
        store = NetworkXGraphStore(persist_path="/tmp/nonexistent_graph.json")
        assert store._graph.number_of_nodes() == 0

    def test_save_without_persist_path(self, graph_store, sample_entities):
        """测试没有持久化路径时的保存。"""
        graph_store.add_entities(sample_entities)
        # 不应该抛出异常
        graph_store.save()

    def test_empty_graph_stats(self, graph_store):
        """测试空图的统计信息。"""
        stats = graph_store.get_stats()
        assert stats["total_entities"] == 0
        assert stats["total_relations"] == 0
        assert stats["documents_count"] == 0
