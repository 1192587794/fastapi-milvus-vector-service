"""
知识图谱业务服务模块。

本模块是知识图谱系统的核心业务层，负责编排实体抽取、关系抽取和图存储，
提供完整的图谱构建、查询、删除能力。

架构位置：
    +------------------+
    |   API 路由层     |  <-- api/graph_routes.py
    +------------------+
            |
            v
    +------------------+
    |   GraphService   |  <-- 本模块（业务逻辑层）
    +------------------+
            |
    +-------+-------+-------+
    |               |       |
    v               v       v
+--------+   +----------+ +--------+
|抽取器  |   |图存储    | |Milvus  |  <-- 基础设施层
+--------+   +----------+ +--------+

GraphService 的职责：
1. 图谱构建：文档文本 --> 实体抽取 --> 关系抽取 --> 写入图存储
2. 图谱查询：用户问题 --> 实体匹配 --> 多跳遍历 --> 返回结果
3. 图谱删除：按文档 ID 清理实体和关系
4. 图谱统计：返回实体/关系的数量和类型分布
5. 图谱可视化：返回前端可用的节点和边数据
6. 上下文生成：将图谱信息格式化为 LLM 可理解的文本

与其他模块的关系：
- EntityExtractor：负责从文本中抽取实体
- RelationExtractor：负责从文本中抽取关系
- GraphStoreProtocol：负责图谱数据的持久化存储
- MilvusManager：负责文档分片的存储和检索（可选）
"""

import logging
from typing import Any

from schemas.graph import Entity, Relation, SubgraphEdge, SubgraphNode
from utils.entity_extractor import EntityExtractor
from utils.relation_extractor import RelationExtractor

logger = logging.getLogger(__name__)


class GraphService:
    """
    知识图谱业务服务。

    这是知识图谱系统的核心类，协调各个组件完成图谱的全生命周期管理。

    典型使用场景：

    1. 文档入库时构建图谱：
       graph_service.build_graph_from_document("doc1", "患者有高血压...")

    2. 用户提问时查询图谱：
       result = graph_service.query_graph("高血压怎么治疗？")

    3. 文档删除时清理图谱：
       graph_service.delete_graph_for_doc("doc1")

    4. 前端可视化时获取子图：
       nodes, edges = graph_service.get_subgraph("高血压", depth=2)

    5. 生成 LLM 上下文：
       context = graph_service.get_relation_context(entities, relations)
    """

    def __init__(
        self,
        settings,
        graph_store,
        entity_extractor: EntityExtractor,
        relation_extractor: RelationExtractor,
        milvus_manager=None,
        embedding_model=None,
    ):
        """
        初始化图谱服务。

        Args:
            settings: Settings 配置对象
            graph_store: 图存储后端（NetworkXGraphStore 或 Neo4jGraphStore）
            entity_extractor: 实体抽取器
            relation_extractor: 关系抽取器
            milvus_manager: Milvus 管理器（可选，用于获取文档分片文本）
            embedding_model: Embedding 模型（可选，用于实体向量化）
        """
        self._settings = settings
        self._store = graph_store
        self._entity_extractor = entity_extractor
        self._relation_extractor = relation_extractor
        self._milvus = milvus_manager
        self._embedding = embedding_model

    def build_graph_from_document(
        self, doc_id: str, text: str, chunk_ids: list[str] | None = None
    ) -> tuple[int, int]:
        """
        从文档文本构建知识图谱。

        这是图谱构建的核心方法，处理流程：
        1. 删除该文档的旧图谱数据（避免重复）
        2. 实体抽取：从文本中提取所有实体
        3. 关系抽取：从文本中提取实体间的关系
        4. 写入图存储：将实体和关系持久化

        为什么先删除旧数据？
        - 避免重复：同一个文档重新上传时，不会产生重复的实体和关系
        - 保证一致性：文档内容更新后，图谱也能同步更新
        - 简化逻辑：不需要复杂的合并和更新策略

        Args:
            doc_id: 文档 ID
            text: 文档文本内容
            chunk_ids: 关联的分片 ID 列表（可选，用于追溯）

        Returns:
            (实体数量, 关系数量) 的元组
        """
        logger.info("Building knowledge graph for document %s", doc_id)

        # 先删除该文档的旧图谱数据
        self.delete_graph_for_doc(doc_id)

        # 实体抽取
        entities = self._entity_extractor.extract(text, doc_id)
        if not entities:
            logger.info("No entities extracted from document %s", doc_id)
            return 0, 0

        # 关系抽取（需要至少2个实体）
        relations = self._relation_extractor.extract(text, entities, doc_id)

        # 写入图存储
        added_entities = self._store.add_entities(entities)
        added_relations = self._store.add_relations(relations)

        # 持久化到磁盘
        self._store.save()

        logger.info(
            "Graph built for doc %s: %d entities (%d new), %d relations (%d new)",
            doc_id, len(entities), added_entities, len(relations), added_relations,
        )
        return len(entities), len(relations)

    def query_graph(
        self, question: str, max_hops: int = 2, top_k: int = 10
    ) -> dict[str, Any]:
        """
        基于问题查询知识图谱。

        这是 GraphRAG 的核心查询方法，处理流程：
        1. 从问题中提取实体（如"高血压"、"头痛"）
        2. 在图谱中查找匹配的实体
        3. 沿着关系边进行多跳遍历
        4. 收集关联的文档分片 ID
        5. 返回实体、关系和分片 ID

        查询示例：
        问题："高血压会导致什么症状？"
        1. 抽取实体：[高血压]
        2. 图谱匹配：找到"高血压"节点
        3. 多跳遍历：
           - 1跳：高血压 --(causes)--> 头痛
           - 2跳：高血压 --(causes)--> 头痛 --(symptom_of)--> 感冒
        4. 返回：实体列表、关系列表、关联的分片 ID

        Args:
            question: 用户问题
            max_hops: 最大跳数（默认2）
            top_k: 返回结果数量上限（默认10）

        Returns:
            包含以下字段的字典：
            - entities: 匹配到的实体列表
            - relations: 实体间的关系列表
            - chunk_ids: 关联的文档分片 ID 列表
        """
        # 从问题中提取实体
        question_entities = self._entity_extractor.extract(question, doc_id="query")

        all_entities = []
        all_relations = []
        chunk_ids = set()

        for qe in question_entities:
            # 在图谱中查找匹配的实体
            matched = self._store.query_entity(qe.name, fuzzy=True)

            for entity in matched:
                # 多跳遍历
                neighbors, relations = self._store.query_neighbors(
                    entity.id, max_hops=max_hops
                )
                all_entities.extend(neighbors)
                all_relations.extend(relations)

                # 收集关联的 chunk_id
                for e in neighbors:
                    if e.chunk_id:
                        chunk_ids.add(e.chunk_id)
                for r in relations:
                    if r.chunk_id:
                        chunk_ids.add(r.chunk_id)

        # 去重
        seen_entities = set()
        unique_entities = []
        for e in all_entities:
            if e.id not in seen_entities:
                seen_entities.add(e.id)
                unique_entities.append(e)

        seen_relations = set()
        unique_relations = []
        for r in all_relations:
            key = (r.source_id, r.target_id, r.relation_type)
            if key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append(r)

        # 限制数量
        unique_entities = unique_entities[:top_k]
        unique_relations = unique_relations[:top_k]

        return {
            "entities": unique_entities,
            "relations": unique_relations,
            "chunk_ids": list(chunk_ids)[:top_k],
        }

    def delete_graph_for_doc(self, doc_id: str) -> tuple[int, int]:
        """
        删除指定文档的图谱数据。

        当文档被删除或重新上传时调用此方法。

        Args:
            doc_id: 要删除的文档 ID

        Returns:
            (删除的实体数, 删除的关系数)
        """
        deleted_entities, deleted_relations = self._store.delete_by_doc(doc_id)

        if deleted_entities or deleted_relations:
            self._store.save()

        return deleted_entities, deleted_relations

    def get_stats(self) -> dict[str, Any]:
        """
        获取图谱统计信息。

        返回信息可用于：
        1. 仪表盘展示图谱规模
        2. 监控图谱构建进度
        3. 分析实体/关系类型分布
        """
        return self._store.get_stats()

    def get_entity_neighbors(
        self, entity_name: str, depth: int = 1
    ) -> tuple[list[Entity], list[Relation]]:
        """
        查询实体的邻居。

        Args:
            entity_name: 实体名称（支持模糊匹配）
            depth: 遍历深度

        Returns:
            (实体列表, 关系列表) 的元组
        """
        entities = self._store.query_entity(entity_name, fuzzy=True)
        if not entities:
            return [], []

        all_neighbors = []
        all_relations = []
        for entity in entities:
            neighbors, relations = self._store.query_neighbors(
                entity.id, max_hops=depth
            )
            all_neighbors.extend(neighbors)
            all_relations.extend(relations)

        return all_neighbors, all_relations

    def get_subgraph(
        self, center_name: str | None = None, depth: int = 1
    ) -> tuple[list[SubgraphNode], list[SubgraphEdge]]:
        """
        获取子图数据，用于前端可视化。

        Args:
            center_name: 中心实体名称（可选）
            depth: 遍历深度

        Returns:
            (节点列表, 边列表) 的元组
        """
        return self._store.get_subgraph(center_name, depth)

    def get_relation_context(
        self, entities: list[Entity], relations: list[Relation]
    ) -> str:
        """
        将实体和关系格式化为文本上下文，用于注入 LLM 提示词。

        这是 GraphRAG 的关键——将图谱信息转换为 LLM 能理解的文本格式。
        生成的文本会被添加到 RAG 的参考资料中，增强 LLM 的回答能力。

        输出示例：
        ```
        相关知识图谱信息：
        [阿司匹林] --(treats)--> [头痛]
        [高血压] --(causes)--> [头痛]
        [头痛] --(symptom_of)--> [感冒]
        ```

        为什么需要这个方法？
        - LLM 无法直接理解图结构数据
        - 需要将图谱信息转换为自然语言描述
        - 这样 LLM 就能利用图谱中的结构化知识来回答问题

        Args:
            entities: 实体列表
            relations: 关系列表

        Returns:
            格式化的文本上下文。如果没有实体和关系，返回空字符串。
        """
        if not entities and not relations:
            return ""

        # 构建实体 ID 到名称的映射
        entity_names = {e.id: e.name for e in entities}

        lines = ["相关知识图谱信息："]

        # 格式化关系
        for rel in relations:
            source_name = entity_names.get(rel.source_id, rel.source_id)
            target_name = entity_names.get(rel.target_id, rel.target_id)
            lines.append(f"[{source_name}] --({rel.relation_type})--> [{target_name}]")

        # 如果只有实体没有关系，也列出实体
        if not relations:
            for e in entities[:10]:
                lines.append(f"[{e.name}]（{e.type}）")

        return "\n".join(lines)
