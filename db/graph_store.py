"""
知识图谱存储抽象层模块。

本模块提供了知识图谱的持久化存储能力，采用策略模式设计：
1. 定义统一的接口协议（GraphStoreProtocol）
2. 提供两种实现：NetworkX（内存）和 Neo4j（生产）
3. 通过工厂函数创建具体的存储实例

架构设计说明：
    +-------------------+
    |   GraphService    |  <-- 业务层
    +-------------------+
            |
            v
    +-------------------+
    | GraphStoreProtocol|  <-- 抽象接口
    +-------------------+
            |
    +-------+-------+
    |               |
    v               v
+--------+    +--------+
|NetworkX|    | Neo4j  |  <-- 具体实现
+--------+    +--------+

为什么需要这个抽象层？
1. 解耦：业务逻辑不依赖具体的存储实现
2. 可替换：可以轻松切换存储后端（从 NetworkX 切换到 Neo4j）
3. 可测试：测试时可以使用内存存储，不需要真实的数据库
4. 配置驱动：通过配置文件选择使用哪种存储后端

NetworkX vs Neo4j 对比：
+---------------+------------------------+------------------------+
| 特性          | NetworkX               | Neo4j                  |
+---------------+------------------------+------------------------+
| 部署方式      | 零依赖，纯 Python      | 需要 Docker 安装       |
| 数据持久化    | JSON 文件              | 数据库原生持久化       |
| 性能          | 适合中小规模（<10万节点）| 适合大规模（百万节点） |
| 查询能力      | Python 代码遍历        | Cypher 图查询语言      |
| 适用场景      | 开发测试、小规模应用    | 生产环境、大规模图谱   |
| 简历价值      | 基础能力               | 加分项                 |
+---------------+------------------------+------------------------+

使用建议：
- 开发和测试阶段使用 NetworkX，零配置即可运行
- 生产环境如果图谱规模较大，切换到 Neo4j
- 通过配置 GRAPH_STORE_BACKEND=networkx/neo4j 控制
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import networkx as nx

from schemas.graph import (
    Entity,
    Relation,
    SubgraphEdge,
    SubgraphNode,
)

logger = logging.getLogger(__name__)


class GraphStoreProtocol(Protocol):
    """
    图存储协议，定义所有图存储后端必须实现的接口。

    这是一个 Protocol 类（结构化子类型），而不是 ABC（抽象基类）。
    Python 的 Protocol 采用鸭子类型（Duck Typing）：
    只要一个类实现了这些方法，就被认为是 GraphStoreProtocol 的实现，
    无论它是否显式继承这个协议。

    这种设计的好处：
    1. 更灵活：不需要修改现有类就能适配协议
    2. 更 Pythonic：符合 Python 的鸭子类型哲学
    3. 更易测试：可以轻松创建 mock 对象

    所有方法的说明见具体实现类中的文档字符串。
    """

    def add_entities(self, entities: list[Entity]) -> int:
        """批量添加实体，返回实际添加数量（去重后）。"""
        ...

    def add_relations(self, relations: list[Relation]) -> int:
        """批量添加关系，返回实际添加数量（去重后）。"""
        ...

    def query_entity(self, name: str, fuzzy: bool = True) -> list[Entity]:
        """按名称查询实体，支持模糊匹配。"""
        ...

    def query_neighbors(
        self, entity_id: str, max_hops: int = 1
    ) -> tuple[list[Entity], list[Relation]]:
        """查询实体的邻居（多跳遍历），返回 (实体列表, 关系列表)。"""
        ...

    def get_subgraph(
        self, center_name: str | None = None, depth: int = 1
    ) -> tuple[list[SubgraphNode], list[SubgraphEdge]]:
        """获取子图，用于可视化。center_name 为空时返回全图摘要。"""
        ...

    def delete_by_doc(self, doc_id: str) -> tuple[int, int]:
        """删除指定文档的所有实体和关系，返回 (删除实体数, 删除关系数)。"""
        ...

    def get_stats(self) -> dict[str, Any]:
        """获取图谱统计信息。"""
        ...

    def save(self) -> None:
        """持久化图谱数据到磁盘。"""
        ...

    def load(self) -> None:
        """从磁盘加载图谱数据。"""
        ...


class NetworkXGraphStore:
    """
    基于 NetworkX 的内存图存储实现。

    NetworkX 是 Python 中最流行的图计算库，提供了丰富的图算法。
    这里使用它的有向图（DiGraph）来存储知识图谱。

    数据结构说明：
    - 节点（Node）= 实体（Entity）
      - 节点属性：name, type, attributes, doc_id, chunk_id
    - 边（Edge）= 关系（Relation）
      - 边属性：relation_type, confidence, doc_id, chunk_id

    为什么使用有向图而不是无向图？
    1. 关系有方向性：[阿司匹林] --(treats)--> [头痛]，方向很重要
    2. 可以遍历出边和入边：查找"谁治疗了X"和"X治疗了谁"
    3. 语义更精确：无向图会丢失方向信息

    持久化机制：
    - 使用 JSON 格式保存到文件
    - NetworkX 提供了 node_link_data/node_link_graph 函数
      可以将图序列化为 JSON 友好的字典格式
    - 服务重启后可以从文件恢复图谱数据

    性能特点：
    - 所有数据存储在内存中，读写速度极快
    - 适合中小规模图谱（<10万节点）
    - 大规模图谱建议使用 Neo4j
    """

    def __init__(self, persist_path: str | None = None):
        """
        初始化 NetworkX 图存储。

        Args:
            persist_path: 图谱数据的持久化文件路径（JSON 格式）
                         如果为 None，则不持久化，数据仅存在于内存中
                         服务重启后数据会丢失
        """
        # 创建有向图实例
        self._graph = nx.DiGraph()
        self._persist_path = persist_path

        # 如果指定了持久化路径，尝试从文件加载已有数据
        if persist_path:
            self.load()

    def add_entities(self, entities: list[Entity]) -> int:
        """
        批量添加实体到图中。

        处理逻辑：
        1. 遍历所有实体
        2. 如果实体不存在（新实体），添加节点并计数
        3. 如果实体已存在（重复实体），更新属性但不计数
        4. 返回实际新增的实体数量

        为什么需要去重？
        - 同一个文档的不同分片可能提到同一个实体
        - 不同文档也可能提到同一个实体
        - 去重可以避免图中出现重复节点

        Args:
            entities: 要添加的实体列表

        Returns:
            实际新增的实体数量（不包括已存在的）
        """
        added = 0
        for entity in entities:
            if not self._graph.has_node(entity.id):
                # 新实体：添加节点及其所有属性
                self._graph.add_node(
                    entity.id,
                    name=entity.name,
                    type=entity.type,
                    attributes=entity.attributes,
                    doc_id=entity.doc_id,
                    chunk_id=entity.chunk_id,
                )
                added += 1
            else:
                # 已存在的实体：只更新属性（保留最新的信息）
                self._graph.nodes[entity.id].update(
                    name=entity.name,
                    type=entity.type,
                    attributes=entity.attributes,
                )
        return added

    def add_relations(self, relations: list[Relation]) -> int:
        """
        批量添加关系到图中。

        处理逻辑与 add_entities 类似：
        1. 如果边不存在，添加边并计数
        2. 如果边已存在，更新属性但不计数

        注意：添加关系前，确保源实体和目标实体已经存在于图中。
        NetworkX 会自动处理这种情况：如果节点不存在，会自动创建。

        Args:
            relations: 要添加的关系列表

        Returns:
            实际新增的关系数量
        """
        added = 0
        for rel in relations:
            if not self._graph.has_edge(rel.source_id, rel.target_id):
                # 新关系：添加边及其属性
                self._graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    relation_type=rel.relation_type,
                    confidence=rel.confidence,
                    doc_id=rel.doc_id,
                    chunk_id=rel.chunk_id,
                )
                added += 1
            else:
                # 已存在的关系：更新属性
                self._graph[rel.source_id][rel.target_id].update(
                    relation_type=rel.relation_type,
                    confidence=rel.confidence,
                )
        return added

    def query_entity(self, name: str, fuzzy: bool = True) -> list[Entity]:
        """
        按名称查询实体。

        支持两种匹配模式：
        1. 精确匹配（fuzzy=False）：名称必须完全一致
        2. 模糊匹配（fuzzy=True）：名称包含查询字符串（不区分大小写）

        模糊匹配的使用场景：
        - 用户输入"血压"，可以匹配到"高血压"
        - 用户输入"阿司匹"，可以匹配到"阿司匹林"
        - 更宽松的匹配，提高召回率

        Args:
            name: 要查询的实体名称
            fuzzy: 是否使用模糊匹配

        Returns:
            匹配到的实体列表
        """
        results = []
        name_lower = name.lower()

        # 遍历图中的所有节点
        for node_id, data in self._graph.nodes(data=True):
            node_name = data.get("name", "")

            if fuzzy:
                # 模糊匹配：双向包含检查
                # "血压" in "高血压" = True
                # "高血压" in "血压" = False
                if name_lower in node_name.lower() or node_name.lower() in name_lower:
                    results.append(self._node_to_entity(node_id, data))
            else:
                # 精确匹配：完全相等
                if node_name == name:
                    results.append(self._node_to_entity(node_id, data))

        return results

    def query_neighbors(
        self, entity_id: str, max_hops: int = 1
    ) -> tuple[list[Entity], list[Relation]]:
        """
        查询实体的邻居（多跳遍历）。

        这是知识图谱最核心的查询能力之一。
        给定一个起始实体，沿着关系边遍历指定的跳数，收集所有可达的实体和关系。

        遍历算法：BFS（广度优先搜索）
        为什么用 BFS 而不是 DFS？
        1. BFS 天然按距离排序：先访问距离为1的节点，再访问距离为2的节点
        2. 更容易控制遍历深度
        3. 在社交网络分析中，BFS 更符合"一度好友"、"二度好友"的概念

        遍历过程示例（max_hops=2）：
        起始实体：阿司匹林

        第1跳：
        阿司匹林 --(treats)--> 头痛
        阿司匹林 --(treats)--> 发热

        第2跳：
        头痛 --(symptom_of)--> 感冒
        发热 --(symptom_of)--> 流感

        最终返回：5个实体 + 4个关系

        Args:
            entity_id: 起始实体的 ID
            max_hops: 最大跳数（默认1跳，即直接邻居）

        Returns:
            (实体列表, 关系列表) 的元组
        """
        # 如果起始实体不存在，返回空结果
        if not self._graph.has_node(entity_id):
            return [], []

        # BFS 遍历所需的集合
        visited_nodes = {entity_id}  # 已访问的节点
        visited_edges = set()         # 已访问的边
        frontier = {entity_id}        # 当前层的节点（待扩展）

        # 逐层遍历
        for _ in range(max_hops):
            next_frontier = set()  # 下一层的节点

            for node in frontier:
                # 遍历出边（node --> target）
                for _, target, data in self._graph.out_edges(node, data=True):
                    edge_key = (node, target)
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                    if target not in visited_nodes:
                        visited_nodes.add(target)
                        next_frontier.add(target)

                # 遍历入边（source --> node）
                for source, _, data in self._graph.in_edges(node, data=True):
                    edge_key = (source, node)
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                    if source not in visited_nodes:
                        visited_nodes.add(source)
                        next_frontier.add(source)

            # 更新前沿为下一层
            frontier = next_frontier

        # 将节点和边转换为 Entity 和 Relation 对象
        entities = [
            self._node_to_entity(nid, self._graph.nodes[nid])
            for nid in visited_nodes
        ]
        relations = [
            self._edge_to_relation(src, tgt, self._graph[src][tgt])
            for src, tgt in visited_edges
        ]

        return entities, relations

    def get_subgraph(
        self, center_name: str | None = None, depth: int = 1
    ) -> tuple[list[SubgraphNode], list[SubgraphEdge]]:
        """
        获取子图数据，用于前端可视化。

        两种模式：
        1. 指定中心实体：以该实体为中心，返回指定跳数内的子图
        2. 不指定中心实体：返回全图摘要（限制节点数量避免过大）

        Args:
            center_name: 中心实体的名称（支持模糊匹配）
                        为 None 时返回全图摘要
            depth: 遍历深度

        Returns:
            (节点列表, 边列表) 的元组，可直接用于前端图可视化库
        """
        if center_name:
            # 模式1：以指定实体为中心
            matches = self.query_entity(center_name, fuzzy=True)
            if not matches:
                return [], []
            center_id = matches[0].id
            entities, relations = self.query_neighbors(center_id, max_hops=depth)
        else:
            # 模式2：全图摘要（限制节点数量避免过大）
            all_nodes = list(self._graph.nodes(data=True))
            if len(all_nodes) > 200:
                all_nodes = all_nodes[:200]  # 限制最多200个节点

            entities = [self._node_to_entity(nid, data) for nid, data in all_nodes]
            relations = [
                self._edge_to_relation(src, tgt, data)
                for src, tgt, data in self._graph.edges(data=True)
                if self._graph.has_node(src) and self._graph.has_node(tgt)
            ][:500]  # 限制最多500条边

        # 转换为前端友好的格式
        nodes = [
            SubgraphNode(
                id=e.id,
                name=e.name,
                type=e.type,
                attributes=e.attributes,
            )
            for e in entities
        ]
        edges = [
            SubgraphEdge(
                source=r.source_id,
                target=r.target_id,
                relation_type=r.relation_type,
                confidence=r.confidence,
            )
            for r in relations
        ]

        return nodes, edges

    def delete_by_doc(self, doc_id: str) -> tuple[int, int]:
        """
        删除指定文档的所有实体和关系。

        当文档被删除时，需要同步清理该文档在知识图谱中的所有数据。
        否则会导致"孤立节点"——指向已不存在的文档。

        Args:
            doc_id: 要删除的文档 ID

        Returns:
            (删除的实体数, 删除的关系数)
        """
        # 找到该文档的所有实体
        nodes_to_remove = [
            nid
            for nid, data in self._graph.nodes(data=True)
            if data.get("doc_id") == doc_id
        ]

        # 找到该文档的所有关系
        edges_to_remove = [
            (src, tgt)
            for src, tgt, data in self._graph.edges(data=True)
            if data.get("doc_id") == doc_id
        ]

        deleted_entities = len(nodes_to_remove)
        deleted_relations = len(edges_to_remove)

        # 先删边再删节点（避免引用错误）
        self._graph.remove_edges_from(edges_to_remove)
        self._graph.remove_nodes_from(nodes_to_remove)

        return deleted_entities, deleted_relations

    def get_stats(self) -> dict[str, Any]:
        """
        获取图谱统计信息。

        返回信息包括：
        - 实体总数和关系总数
        - 各类型实体的数量分布
        - 各类型关系的数量分布
        - 涉及的文档数量

        这些信息可用于：
        1. 仪表盘展示图谱规模
        2. 监控图谱构建进度
        3. 分析实体/关系类型分布是否合理
        """
        # 统计实体类型分布
        entity_type_counts = Counter()
        for _, data in self._graph.nodes(data=True):
            entity_type_counts[data.get("type", "Other")] += 1

        # 统计关系类型分布
        relation_type_counts = Counter()
        for _, _, data in self._graph.edges(data=True):
            relation_type_counts[data.get("relation_type", "unknown")] += 1

        # 统计涉及的文档数量
        doc_ids = set()
        for _, data in self._graph.nodes(data=True):
            doc_ids.add(data.get("doc_id", ""))

        return {
            "total_entities": self._graph.number_of_nodes(),
            "total_relations": self._graph.number_of_edges(),
            "entity_type_counts": dict(entity_type_counts),
            "relation_type_counts": dict(relation_type_counts),
            "documents_count": len(doc_ids),
        }

    def save(self) -> None:
        """
        持久化图谱数据到 JSON 文件。

        使用 NetworkX 的 node_link_data 函数将图序列化为字典，
        然后写入 JSON 文件。

        JSON 文件格式示例：
        {
            "directed": true,
            "multigraph": false,
            "graph": {},
            "nodes": [
                {"id": "doc1::entity::阿司匹林::Drug", "name": "阿司匹林", ...},
                ...
            ],
            "links": [
                {"source": "...", "target": "...", "relation_type": "treats", ...},
                ...
            ]
        }
        """
        if not self._persist_path:
            return

        path = Path(self._persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 将图转换为可序列化的字典
        data = nx.node_link_data(self._graph)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info(
            "Graph saved to %s (%d nodes, %d edges)",
            path,
            self._graph.number_of_nodes(),
            self._graph.number_of_edges()
        )

    def load(self) -> None:
        """
        从 JSON 文件加载图谱数据。

        如果文件不存在或加载失败，会创建一个空图并记录日志。
        这种设计保证了服务的健壮性——即使图谱文件损坏也不会导致服务崩溃。
        """
        if not self._persist_path:
            return

        path = Path(self._persist_path)
        if not path.exists():
            logger.info("No persisted graph found at %s, starting with empty graph", path)
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # 从字典恢复图结构
            self._graph = nx.node_link_graph(data, directed=True)
            logger.info(
                "Graph loaded from %s (%d nodes, %d edges)",
                path,
                self._graph.number_of_nodes(),
                self._graph.number_of_edges()
            )
        except Exception:
            logger.warning(
                "Failed to load graph from %s, starting with empty graph",
                path,
                exc_info=True
            )
            self._graph = nx.DiGraph()

    def _node_to_entity(self, node_id: str, data: dict) -> Entity:
        """
        将 NetworkX 节点转换为 Entity 对象。

        这是一个内部辅助方法，用于统一节点数据的转换逻辑。
        """
        return Entity(
            id=node_id,
            name=data.get("name", ""),
            type=data.get("type", "Other"),
            attributes=data.get("attributes", {}),
            doc_id=data.get("doc_id", ""),
            chunk_id=data.get("chunk_id"),
        )

    def _edge_to_relation(self, source_id: str, target_id: str, data: dict) -> Relation:
        """
        将 NetworkX 边转换为 Relation 对象。

        这是一个内部辅助方法，用于统一边数据的转换逻辑。
        """
        return Relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=data.get("relation_type", "unknown"),
            confidence=data.get("confidence", 1.0),
            doc_id=data.get("doc_id", ""),
            chunk_id=data.get("chunk_id"),
        )


class Neo4jGraphStore:
    """
    基于 Neo4j 的图数据库存储实现。

    Neo4j 是最流行的图数据库，专为存储和查询图结构数据而设计。
    相比 NetworkX，Neo4j 的优势：
    1. 持久化：数据存储在磁盘上，服务重启不丢失
    2. 高性能：针对图查询优化，百万节点也能快速遍历
    3. Cypher 查询语言：声明式的图查询语言，比 Python 代码更简洁
    4. 事务支持：保证数据一致性
    5. 可视化工具：Neo4j Browser 提供图形化界面

    安装和使用：
    1. 安装 Neo4j：
       docker run -d -p 7474:7474 -p 7687:7687 neo4j
    2. 安装 Python 驱动：
       pip install neo4j
    3. 配置环境变量：
       NEO4J_URI=bolt://localhost:7687
       NEO4J_USER=neo4j
       NEO4J_PASSWORD=your_password

    Neo4j 中的数据模型：
    - 节点标签（Label）：Entity
    - 节点属性：id, name, type, attributes, doc_id, chunk_id
    - 关系类型（Relationship Type）：RELATED
    - 关系属性：relation_type, confidence, doc_id, chunk_id

    注意：这个类需要安装 neo4j 驱动才能使用。
    如果未安装，会在初始化时抛出 ImportError。
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        初始化 Neo4j 图存储。

        Args:
            uri: Neo4j 连接地址，如 "bolt://localhost:7687"
            user: 用户名，默认 "neo4j"
            password: 密码
            database: 数据库名称，默认 "neo4j"

        Raises:
            ImportError: 如果未安装 neo4j 驱动
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "Neo4j backend requires the 'neo4j' package. "
                "Install it with: pip install neo4j"
            )

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        """
        确保数据库中存在必要的约束。

        约束的作用：
        1. 唯一性约束：保证实体 ID 不重复
        2. 索引：加速按 ID 查询的速度
        """
        with self._driver.session(database=self._database) as session:
            session.run(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            )

    def add_entities(self, entities: list[Entity]) -> int:
        """
        批量添加实体到 Neo4j。

        使用 Cypher 的 UNWIND + MERGE 语法：
        - UNWIND：将列表展开为多行
        - MERGE：如果节点存在则匹配，不存在则创建
        - SET：更新节点属性

        这种写法比逐条插入高效得多，因为只需要一次数据库往返。
        """
        query = """
        UNWIND $entities AS e
        MERGE (n:Entity {id: e.id})
        SET n.name = e.name,
            n.type = e.type,
            n.attributes = e.attributes,
            n.doc_id = e.doc_id,
            n.chunk_id = e.chunk_id
        RETURN count(n) AS added
        """
        entity_data = [e.model_dump() for e in entities]
        with self._driver.session(database=self._database) as session:
            result = session.run(query, entities=entity_data)
            return result.single()["added"]

    def add_relations(self, relations: list[Relation]) -> int:
        """
        批量添加关系到 Neo4j。

        使用 MATCH 找到源实体和目标实体，然后 MERGE 关系。
        """
        query = """
        UNWIND $relations AS r
        MATCH (a:Entity {id: r.source_id})
        MATCH (b:Entity {id: r.target_id})
        MERGE (a)-[rel:RELATED {source_id: r.source_id, target_id: r.target_id}]->(b)
        SET rel.relation_type = r.relation_type,
            rel.confidence = r.confidence,
            rel.doc_id = r.doc_id,
            rel.chunk_id = r.chunk_id
        RETURN count(rel) AS added
        """
        rel_data = [r.model_dump() for r in relations]
        with self._driver.session(database=self._database) as session:
            result = session.run(query, relations=rel_data)
            return result.single()["added"]

    def query_entity(self, name: str, fuzzy: bool = True) -> list[Entity]:
        """
        按名称查询实体。

        CONTAINS 是 Cypher 的字符串包含操作符。
        """
        if fuzzy:
            query = "MATCH (e:Entity) WHERE e.name CONTAINS $name RETURN e"
        else:
            query = "MATCH (e:Entity {name: $name}) RETURN e"

        with self._driver.session(database=self._database) as session:
            result = session.run(query, name=name)
            return [
                Entity(
                    id=record["e"]["id"],
                    name=record["e"]["name"],
                    type=record["e"]["type"],
                    attributes=record["e"].get("attributes", {}),
                    doc_id=record["e"].get("doc_id", ""),
                    chunk_id=record["e"].get("chunk_id"),
                )
                for record in result
            ]

    def query_neighbors(
        self, entity_id: str, max_hops: int = 1
    ) -> tuple[list[Entity], list[Relation]]:
        """
        查询实体的邻居（多跳遍历）。

        使用 Cypher 的可变长度路径语法：[*0..N]
        表示匹配0到N跳的路径。
        """
        entity_query = """
        MATCH (e:Entity {id: $entity_id})-[*0..""" + str(max_hops) + """]-(neighbor)
        RETURN DISTINCT neighbor
        """
        relation_query = """
        MATCH (e:Entity {id: $entity_id})-[*0..""" + str(max_hops) + """]-(neighbor)
        MATCH (a)-[rel:RELATED]-(b)
        WHERE a IN collect(neighbor) + [e] AND b IN collect(neighbor) + [e]
        RETURN DISTINCT rel, a, b
        """

        with self._driver.session(database=self._database) as session:
            # 查询实体
            entity_result = session.run(entity_query, entity_id=entity_id)
            entities = []
            for record in entity_result:
                node = record["neighbor"]
                entities.append(Entity(
                    id=node["id"],
                    name=node["name"],
                    type=node["type"],
                    attributes=node.get("attributes", {}),
                    doc_id=node.get("doc_id", ""),
                    chunk_id=node.get("chunk_id"),
                ))

            # 查询关系
            rel_result = session.run(relation_query, entity_id=entity_id)
            relations = []
            for record in rel_result:
                rel = record["rel"]
                relations.append(Relation(
                    source_id=record["a"]["id"],
                    target_id=record["b"]["id"],
                    relation_type=rel["relation_type"],
                    confidence=rel["confidence"],
                    doc_id=rel.get("doc_id", ""),
                    chunk_id=rel.get("chunk_id"),
                ))

        return entities, relations

    def get_subgraph(
        self, center_name: str | None = None, depth: int = 1
    ) -> tuple[list[SubgraphNode], list[SubgraphEdge]]:
        """获取子图数据，用于前端可视化。"""
        if center_name:
            query = """
            MATCH (e:Entity)-[*0..""" + str(depth) + """]-(neighbor)
            WHERE e.name CONTAINS $name
            WITH collect(DISTINCT neighbor) + [e] AS nodes
            UNWIND nodes AS n
            OPTIONAL MATCH (n)-[rel:RELATED]-(m)
            WHERE m IN nodes
            RETURN DISTINCT n, rel, m
            """
            with self._driver.session(database=self._database) as session:
                result = session.run(query, name=center_name)
                nodes_dict = {}
                edges = []
                for record in result:
                    n = record["n"]
                    nodes_dict[n["id"]] = SubgraphNode(
                        id=n["id"], name=n["name"], type=n["type"],
                        attributes=n.get("attributes", {}),
                    )
                    if record["rel"] and record["m"]:
                        m = record["m"]
                        nodes_dict[m["id"]] = SubgraphNode(
                            id=m["id"], name=m["name"], type=m["type"],
                            attributes=m.get("attributes", {}),
                        )
                        rel = record["rel"]
                        edges.append(SubgraphEdge(
                            source=record["n"]["id"],
                            target=m["id"],
                            relation_type=rel["relation_type"],
                            confidence=rel["confidence"],
                        ))
                return list(nodes_dict.values()), edges
        else:
            # 全图摘要
            query = """
            MATCH (e:Entity) WITH e LIMIT 200
            OPTIONAL MATCH (e)-[rel:RELATED]-(m)
            RETURN DISTINCT e, rel, m LIMIT 500
            """
            with self._driver.session(database=self._database) as session:
                result = session.run(query)
                nodes_dict = {}
                edges = []
                for record in result:
                    e = record["e"]
                    nodes_dict[e["id"]] = SubgraphNode(
                        id=e["id"], name=e["name"], type=e["type"],
                        attributes=e.get("attributes", {}),
                    )
                    if record["rel"] and record["m"]:
                        m = record["m"]
                        nodes_dict[m["id"]] = SubgraphNode(
                            id=m["id"], name=m["name"], type=m["type"],
                            attributes=m.get("attributes", {}),
                        )
                        rel = record["rel"]
                        edges.append(SubgraphEdge(
                            source=e["id"],
                            target=m["id"],
                            relation_type=rel["relation_type"],
                            confidence=rel["confidence"],
                        ))
                return list(nodes_dict.values()), edges

    def delete_by_doc(self, doc_id: str) -> tuple[int, int]:
        """删除指定文档的所有实体和关系。"""
        with self._driver.session(database=self._database) as session:
            # 删除关系
            rel_result = session.run(
                "MATCH ()-[r:RELATED {doc_id: $doc_id}]-() DELETE r RETURN count(r) AS cnt",
                doc_id=doc_id,
            )
            deleted_relations = rel_result.single()["cnt"]

            # 删除实体
            entity_result = session.run(
                "MATCH (e:Entity {doc_id: $doc_id}) DETACH DELETE e RETURN count(e) AS cnt",
                doc_id=doc_id,
            )
            deleted_entities = entity_result.single()["cnt"]

        return deleted_entities, deleted_relations

    def get_stats(self) -> dict[str, Any]:
        """获取图谱统计信息。"""
        with self._driver.session(database=self._database) as session:
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS cnt").single()["cnt"]
            relation_count = session.run("MATCH ()-[r:RELATED]-() RETURN count(r) AS cnt").single()["cnt"]

            entity_types = session.run(
                "MATCH (e:Entity) RETURN e.type AS type, count(*) AS cnt"
            )
            entity_type_counts = {record["type"]: record["cnt"] for record in entity_types}

            relation_types = session.run(
                "MATCH ()-[r:RELATED]-() RETURN r.relation_type AS type, count(*) AS cnt"
            )
            relation_type_counts = {record["type"]: record["cnt"] for record in relation_types}

            doc_count = session.run(
                "MATCH (e:Entity) RETURN count(DISTINCT e.doc_id) AS cnt"
            ).single()["cnt"]

        return {
            "total_entities": entity_count,
            "total_relations": relation_count,
            "entity_type_counts": entity_type_counts,
            "relation_type_counts": relation_type_counts,
            "documents_count": doc_count,
        }

    def save(self) -> None:
        """
        持久化图谱数据。

        对于 Neo4j，数据由数据库自身管理，不需要额外的持久化操作。
        """
        pass

    def load(self) -> None:
        """
        加载图谱数据。

        对于 Neo4j，数据由数据库自身管理，不需要额外的加载操作。
        """
        pass

    def close(self) -> None:
        """关闭 Neo4j 连接。"""
        self._driver.close()


def create_graph_store(settings) -> Any:
    """
    图存储工厂函数。

    根据配置创建对应的存储后端实例。
    这是工厂模式的典型应用：
    - 调用方不需要知道具体的实现类
    - 只需要通过配置来选择使用哪种后端
    - 新增后端时只需要修改这个函数

    Args:
        settings: Settings 配置对象

    Returns:
        GraphStoreProtocol 的实现实例（NetworkXGraphStore 或 Neo4jGraphStore）
    """
    backend = settings.graph_store_backend.lower()

    if backend == "neo4j":
        return Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
    else:
        # 默认使用 NetworkX
        persist_path = settings.graph_persist_path if settings.graph_persist_path else None
        return NetworkXGraphStore(persist_path=persist_path)
