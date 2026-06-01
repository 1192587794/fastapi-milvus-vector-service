"""
知识图谱相关的数据模型模块。

本模块定义了知识图谱系统中所有的数据结构，包括：
1. 实体（Entity）：知识图谱中的节点，代表具体的医疗概念
2. 关系（Relation）：知识图谱中的边，代表实体之间的语义关系
3. API 请求/响应模型：用于 FastAPI 路由的数据校验和序列化

知识图谱的基本结构：
    实体A --(关系类型)--> 实体B
    例如：[阿司匹林] --(treats)--> [头痛]

医疗领域实体类型说明：
- Disease（疾病）：如高血压、糖尿病、冠心病
- Symptom（症状）：如头痛、发热、胸闷
- Drug（药物）：如阿司匹林、青霉素、胰岛素
- Procedure（手术/操作）：如冠状动脉搭桥术、阑尾切除术
- Department（科室）：如心内科、神经外科、急诊科
- AnatomicalPart（解剖部位）：如心脏、肝脏、大脑
- MedicalDevice（医疗器械）：如呼吸机、心电监护仪
- Other（其他）：不属于以上类型的实体

关系类型说明：
- treats（治疗）：药物/手术治疗疾病
- causes（导致）：疾病/因素导致症状或其他疾病
- symptom_of（症状属于）：症状是某种疾病的表现
- used_for（用于）：药物/设备用于某种治疗
- belongs_to（属于）：科室/人员属于某个部门
- part_of（是...的一部分）：解剖部位是某个器官的一部分
- interacts_with（相互作用）：药物之间或药物与食物的相互作用
- contradicts（禁忌/矛盾）：药物禁忌或治疗矛盾
"""

from typing import Any

from pydantic import BaseModel, Field


# --- 实体类型枚举 ---
# 定义了系统支持的所有实体类型
# 在实体抽取时，LLM 会被要求将文本中的实体分类到这些类型中
ENTITY_TYPES = [
    "Disease",        # 疾病
    "Symptom",        # 症状
    "Drug",           # 药物
    "Procedure",      # 手术/操作
    "Department",     # 科室
    "AnatomicalPart", # 解剖部位
    "MedicalDevice",  # 医疗器械
    "Other",          # 其他
]

# --- 关系类型枚举 ---
# 定义了系统支持的所有关系类型
# 在关系抽取时，LLM 会被要求将实体间的关系分类到这些类型中
RELATION_TYPES = [
    "treats",         # 治疗
    "causes",         # 导致
    "symptom_of",     # 症状属于
    "used_for",       # 用于
    "belongs_to",     # 属于
    "part_of",        # 是...的一部分
    "interacts_with", # 相互作用
    "contradicts",    # 禁忌/矛盾
]


class Entity(BaseModel):
    """
    知识图谱中的实体节点。

    实体是知识图谱的基本组成单元，代表一个具体的医疗概念。
    每个实体都有唯一的标识符、名称、类型和可选的属性。

    实体 ID 的生成规则：
        {doc_id}::entity::{name}::{type}
        例如：doc1::entity::阿司匹林::Drug

    属性说明：
        id: 实体的唯一标识符，由文档ID、实体名称和类型组合而成
        name: 实体的显示名称，如"阿司匹林"、"高血压"
        type: 实体类型，必须是 ENTITY_TYPES 中的一种
        attributes: 实体的扩展属性字典，如{"描述": "抗血小板药物", "剂量": "100mg"}
        doc_id: 来源文档的 ID，用于追溯实体的来源
        chunk_id: 来源分片的 ID，用于定位实体在文档中的具体位置
    """

    id: str = Field(
        ...,
        description="实体唯一标识，格式: {doc_id}::entity::{name}::{type}",
        examples=["doc1::entity::阿司匹林::Drug"]
    )
    name: str = Field(
        ...,
        description="实体名称，如 '阿司匹林'、'高血压'",
        min_length=1,
        max_length=200
    )
    type: str = Field(
        ...,
        description="实体类型，必须是 ENTITY_TYPES 中的一种"
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="实体属性字典，用于存储额外的描述信息"
    )
    doc_id: str = Field(
        ...,
        description="来源文档 ID，用于追溯实体来源"
    )
    chunk_id: str | None = Field(
        default=None,
        description="来源分片 ID，用于定位实体在文档中的具体位置"
    )


class Relation(BaseModel):
    """
    知识图谱中的关系边。

    关系连接两个实体，表示它们之间的语义联系。
    每个关系都有方向性（从源实体指向目标实体）和类型。

    关系的表示方式：
        源实体 --(关系类型)--> 目标实体
        例如：[阿司匹林] --(treats)--> [头痛]

    属性说明：
        source_id: 源实体的 ID（关系的起点）
        target_id: 目标实体的 ID（关系的终点）
        relation_type: 关系类型，必须是 RELATION_TYPES 中的一种
        confidence: 关系的置信度（0-1），表示 LLM 对这个关系判断的把握程度
        doc_id: 来源文档 ID
        chunk_id: 来源分片 ID
    """

    source_id: str = Field(
        ...,
        description="源实体 ID（关系的起点）"
    )
    target_id: str = Field(
        ...,
        description="目标实体 ID（关系的终点）"
    )
    relation_type: str = Field(
        ...,
        description="关系类型，必须是 RELATION_TYPES 中的一种"
    )
    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="关系置信度（0-1），越高表示 LLM 对这个关系越确定"
    )
    doc_id: str = Field(
        ...,
        description="来源文档 ID"
    )
    chunk_id: str | None = Field(
        default=None,
        description="来源分片 ID"
    )


# ============================================================
# API 请求/响应模型
# 以下模型用于 FastAPI 路由的数据校验和序列化
# ============================================================


class GraphBuildRequest(BaseModel):
    """
    手动触发图谱构建的请求模型。

    当用户需要对某个文档重新构建知识图谱时，可以调用此接口。
    通常在以下场景使用：
    1. 文档上传后自动构建失败，需要手动重试
    2. 更新了实体抽取的 prompt，需要重新抽取
    3. 调试和测试图谱构建功能
    """

    doc_id: str = Field(
        ...,
        description="文档 ID，用于标识要构建图谱的文档"
    )
    text: str = Field(
        ...,
        min_length=1,
        description="文档文本内容，将从此文本中抽取实体和关系"
    )


class GraphBuildResponse(BaseModel):
    """
    图谱构建结果的响应模型。

    返回构建过程中抽取到的实体和关系数量，
    用于确认构建是否成功以及抽取效果。
    """

    doc_id: str = Field(description="文档 ID")
    entities_count: int = Field(description="抽取到的实体数量")
    relations_count: int = Field(description="抽取到的关系数量")


class GraphQueryRequest(BaseModel):
    """
    图谱查询请求模型。

    用户输入一个问题或关键词，系统会：
    1. 从查询文本中提取实体
    2. 在知识图谱中查找匹配的实体
    3. 沿着关系边进行多跳遍历
    4. 返回相关的实体、关系和关联的文档分片
    """

    query: str = Field(
        ...,
        min_length=1,
        description="查询文本，用于实体匹配"
    )
    max_hops: int = Field(
        default=2,
        ge=1,
        le=4,
        description="最大跳数，控制图谱遍历的深度。1=直接邻居，2=两跳可达的实体"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="返回结果数量上限"
    )


class GraphQueryResponse(BaseModel):
    """
    图谱查询结果的响应模型。

    包含查询匹配到的实体、它们之间的关系，以及关联的文档分片 ID。
    这些信息可以用于：
    1. 直接展示给用户
    2. 作为 RAG 流水线的图谱召回结果
    3. 注入到 LLM 的提示词中，增强问答能力
    """

    query: str = Field(description="原始查询文本")
    entities: list[Entity] = Field(description="匹配到的实体列表")
    relations: list[Relation] = Field(description="实体之间的关系列表")
    source_chunks: list[str] = Field(
        default_factory=list,
        description="关联的文档分片 ID 列表，可用于获取原始文本"
    )


class SubgraphRequest(BaseModel):
    """
    子图查询请求模型（用于前端可视化）。

    以指定实体为中心，返回指定跳数内的所有节点和边。
    前端可以用这些数据绘制知识图谱的可视化图表。

    使用场景：
    1. 用户点击某个实体，查看其关联的实体和关系
    2. 展示某个疾病的所有相关症状、药物、手术等
    3. 全图概览，了解知识图谱的整体结构
    """

    entity_name: str | None = Field(
        default=None,
        description="中心实体名称。为空时返回全图摘要（限制节点数量）"
    )
    depth: int = Field(
        default=1,
        ge=1,
        le=3,
        description="遍历深度。1=只显示直接邻居，2=显示两跳内的所有实体"
    )


class SubgraphNode(BaseModel):
    """
    子图节点模型（前端可视化用）。

    代表知识图谱中的一个节点，前端可以用这些信息绘制图形。
    """

    id: str = Field(description="节点唯一标识")
    name: str = Field(description="节点显示名称")
    type: str = Field(description="节点类型，用于决定显示样式")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="节点属性，可用于 tooltip 显示"
    )


class SubgraphEdge(BaseModel):
    """
    子图边模型（前端可视化用）。

    代表知识图谱中的一条边（关系），连接两个节点。
    """

    source: str = Field(description="源节点 ID")
    target: str = Field(description="目标节点 ID")
    relation_type: str = Field(description="关系类型，用于边的标签")
    confidence: float = Field(
        default=1.0,
        description="关系置信度，可用于边的粗细或透明度"
    )


class SubgraphResponse(BaseModel):
    """
    子图查询结果的响应模型。

    包含节点列表和边列表，可直接用于前端图可视化库（如 D3.js、ECharts、Vis.js）。
    """

    nodes: list[SubgraphNode] = Field(description="节点列表")
    edges: list[SubgraphEdge] = Field(description="边列表")


class GraphStatsResponse(BaseModel):
    """
    图谱统计信息的响应模型。

    提供知识图谱的整体概况，包括：
    - 实体和关系的总数
    - 各类型实体/关系的数量分布
    - 涉及的文档数量

    可用于：
    1. 仪表盘展示图谱规模
    2. 监控图谱构建进度
    3. 分析实体/关系类型分布
    """

    total_entities: int = Field(description="实体总数")
    total_relations: int = Field(description="关系总数")
    entity_type_counts: dict[str, int] = Field(
        default_factory=dict,
        description="各类型实体的数量，如 {'Disease': 10, 'Drug': 5}"
    )
    relation_type_counts: dict[str, int] = Field(
        default_factory=dict,
        description="各类型关系的数量，如 {'treats': 8, 'causes': 3}"
    )
    documents_count: int = Field(
        default=0,
        description="涉及的文档数量"
    )


class GraphDeleteResponse(BaseModel):
    """
    图谱删除结果的响应模型。

    当删除某个文档时，同时清理该文档在知识图谱中的所有实体和关系。
    返回实际删除的数量，用于确认删除操作的影响范围。
    """

    doc_id: str = Field(description="被删除的文档 ID")
    deleted_entities: int = Field(description="删除的实体数量")
    deleted_relations: int = Field(description="删除的关系数量")
