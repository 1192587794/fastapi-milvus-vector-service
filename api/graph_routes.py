"""
知识图谱管理 API 路由模块。

本模块提供了知识图谱的 REST API 端点，用于：
1. 图谱构建：手动触发图谱构建
2. 图谱统计：获取实体/关系的数量和类型分布
3. 图谱查询：基于问题查询相关实体和关系
4. 子图获取：获取指定实体的邻居图（用于前端可视化）
5. 图谱删除：删除指定文档的图谱数据

API 端点列表：
- POST   /api/v1/graph/build      -- 构建图谱
- GET    /api/v1/graph/stats      -- 获取统计信息
- POST   /api/v1/graph/query      -- 查询图谱
- POST   /api/v1/graph/subgraph   -- 获取子图
- DELETE /api/v1/graph/{doc_id}    -- 删除图谱

使用场景：
1. 手动重建图谱：文档上传后自动构建失败，需要手动重试
2. 查看图谱概况：通过 stats 端点了解图谱规模
3. 调试图谱查询：通过 query 端点测试查询效果
4. 前端可视化：通过 subgraph 端点获取图形数据
5. 清理图谱数据：文档删除后清理关联的图谱数据

前置条件：
- 需要在 .env 中设置 ENABLE_KNOWLEDGE_GRAPH=true
- 如果未启用，所有端点会返回 503 Service Unavailable
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from schemas.graph import (
    GraphBuildRequest,
    GraphBuildResponse,
    GraphDeleteResponse,
    GraphQueryRequest,
    GraphQueryResponse,
    GraphStatsResponse,
    SubgraphRequest,
    SubgraphResponse,
)

logger = logging.getLogger(__name__)

# 创建路由器，所有端点以 /api/v1/graph 为前缀
router = APIRouter(prefix="/api/v1/graph", tags=["Knowledge Graph"])


def _get_graph_service(request: Request):
    """
    获取图谱服务实例的依赖函数。

    这是一个辅助函数，用于从 app.state 中获取图谱服务。
    如果图谱服务未启用（ENABLE_KNOWLEDGE_GRAPH=false），
    会抛出 503 异常。

    为什么需要这个函数？
    - 图谱功能是可选的，不启用时不应该影响其他功能
    - 统一处理"服务未启用"的情况，避免在每个端点中重复检查

    Args:
        request: FastAPI 请求对象

    Returns:
        GraphService 实例

    Raises:
        HTTPException: 如果图谱服务未启用
    """
    graph_service = getattr(request.app.state, "graph_service", None)
    if not graph_service:
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph service is not enabled. Set ENABLE_KNOWLEDGE_GRAPH=true to enable.",
        )
    return graph_service


@router.post("/build", response_model=GraphBuildResponse)
async def build_graph(request: Request, body: GraphBuildRequest):
    """
    手动触发知识图谱构建。

    从指定文档文本中抽取实体和关系，构建知识图谱。
    通常在文档上传时自动触发，此端点用于：
    1. 自动构建失败后的手动重试
    2. 更新抽取策略后的重新构建
    3. 调试和测试

    请求示例：
    {
        "doc_id": "doc1",
        "text": "患者有高血压病史，长期服用阿司匹林100mg..."
    }

    响应示例：
    {
        "doc_id": "doc1",
        "entities_count": 5,
        "relations_count": 3
    }
    """
    graph_service = _get_graph_service(request)

    entities_count, relations_count = graph_service.build_graph_from_document(
        doc_id=body.doc_id, text=body.text
    )

    return GraphBuildResponse(
        doc_id=body.doc_id,
        entities_count=entities_count,
        relations_count=relations_count,
    )


@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats(request: Request):
    """
    获取知识图谱统计信息。

    返回图谱的整体概况，包括：
    - 实体总数和关系总数
    - 各类型实体的数量分布
    - 各类型关系的数量分布
    - 涉及的文档数量

    响应示例：
    {
        "total_entities": 100,
        "total_relations": 80,
        "entity_type_counts": {"Disease": 30, "Drug": 25, "Symptom": 20, ...},
        "relation_type_counts": {"treats": 40, "causes": 30, ...},
        "documents_count": 10
    }
    """
    graph_service = _get_graph_service(request)
    stats = graph_service.get_stats()
    return GraphStatsResponse(**stats)


@router.post("/query", response_model=GraphQueryResponse)
async def query_graph(request: Request, body: GraphQueryRequest):
    """
    查询知识图谱。

    基于问题文本匹配实体，支持多跳遍历返回相关实体和关系。
    这是 GraphRAG 的核心查询能力。

    查询流程：
    1. 从查询文本中提取实体
    2. 在图谱中查找匹配的实体
    3. 沿着关系边进行多跳遍历
    4. 返回相关的实体、关系和关联的文档分片 ID

    请求示例：
    {
        "query": "高血压会导致什么症状？",
        "max_hops": 2,
        "top_k": 10
    }

    响应示例：
    {
        "query": "高血压会导致什么症状？",
        "entities": [
            {"id": "...", "name": "高血压", "type": "Disease", ...},
            {"id": "...", "name": "头痛", "type": "Symptom", ...},
            ...
        ],
        "relations": [
            {"source_id": "...", "target_id": "...", "relation_type": "causes", ...},
            ...
        ],
        "source_chunks": ["doc1::chunk::0", "doc1::chunk::3"]
    }
    """
    graph_service = _get_graph_service(request)

    result = graph_service.query_graph(
        question=body.query,
        max_hops=body.max_hops,
        top_k=body.top_k,
    )

    return GraphQueryResponse(
        query=body.query,
        entities=result.get("entities", []),
        relations=result.get("relations", []),
        source_chunks=result.get("chunk_ids", []),
    )


@router.post("/subgraph", response_model=SubgraphResponse)
async def get_subgraph(request: Request, body: SubgraphRequest):
    """
    获取子图数据（用于前端可视化）。

    以指定实体为中心，返回指定跳数内的所有节点和边。
    前端可以用这些数据绘制知识图谱的可视化图表。

    使用场景：
    1. 用户点击某个实体，查看其关联的实体和关系
    2. 展示某个疾病的所有相关症状、药物、手术等
    3. 全图概览，了解知识图谱的整体结构

    请求示例：
    {
        "entity_name": "高血压",
        "depth": 2
    }

    响应示例：
    {
        "nodes": [
            {"id": "...", "name": "高血压", "type": "Disease", "attributes": {}},
            {"id": "...", "name": "头痛", "type": "Symptom", "attributes": {}},
            ...
        ],
        "edges": [
            {"source": "...", "target": "...", "relation_type": "causes", "confidence": 0.9},
            ...
        ]
    }

    前端可视化建议：
    - 使用 D3.js、ECharts、Vis.js 等图可视化库
    - 节点颜色按实体类型区分（如 Disease=红色，Drug=蓝色）
    - 边的标签显示关系类型
    - 边的粗细或透明度可按置信度调整
    """
    graph_service = _get_graph_service(request)

    nodes, edges = graph_service.get_subgraph(
        center_name=body.entity_name,
        depth=body.depth,
    )

    return SubgraphResponse(nodes=nodes, edges=edges)


@router.delete("/{doc_id}", response_model=GraphDeleteResponse)
async def delete_graph(request: Request, doc_id: str):
    """
    删除指定文档的知识图谱数据。

    当文档被删除时，需要同步清理该文档在知识图谱中的所有实体和关系。
    否则会导致"孤立节点"——指向已不存在的文档。

    响应示例：
    {
        "doc_id": "doc1",
        "deleted_entities": 5,
        "deleted_relations": 3
    }
    """
    graph_service = _get_graph_service(request)

    deleted_entities, deleted_relations = graph_service.delete_graph_for_doc(doc_id)

    return GraphDeleteResponse(
        doc_id=doc_id,
        deleted_entities=deleted_entities,
        deleted_relations=deleted_relations,
    )
