# 知识图谱（Knowledge Graph）学习指南

## 目录

1. [什么是知识图谱](#1-什么是知识图谱)
2. [为什么需要知识图谱](#2-为什么需要知识图谱)
3. [知识图谱 vs 向量检索](#3-知识图谱-vs-向量检索)
4. [项目架构设计](#4-项目架构设计)
5. [核心概念详解](#5-核心概念详解)
6. [代码实现详解](#6-代码实现详解)
7. [GraphRAG 原理](#7-graphrag-原理)
8. [如何使用](#8-如何使用)
9. [简历写法建议](#9-简历写法建议)
10. [进阶学习路线](#10-进阶学习路线)

---

## 1. 什么是知识图谱

### 1.1 基本定义

知识图谱（Knowledge Graph）是一种用**图结构**来表示知识的数据组织方式。

它由三个核心元素组成：
- **实体（Entity）**：现实世界中的事物，如"阿司匹林"、"高血压"
- **关系（Relation）**：实体之间的联系，如"治疗"、"导致"
- **属性（Attribute）**：实体的特征，如"剂量：100mg"

### 1.2 直观理解

想象一个医疗知识网络：

```
[阿司匹林] --(治疗)--> [头痛]
    |
    +-----(导致)--> [胃出血]
    
[高血压] --(导致)--> [头痛]
    |
    +-----(属于)--> [心血管疾病]
    
[头痛] --(症状属于)--> [感冒]
```

这就是一个简单的知识图谱：
- 方括号 `[]` 中的是**实体**
- 箭头 `-->` 表示**关系方向**
- 括号 `()` 中的是**关系类型**

### 1.3 形式化表示

知识图谱通常用**三元组**（Triple）表示：

```
(主语, 谓语, 宾语)
(Subject, Predicate, Object)
```

例如：
- (阿司匹林, treats, 头痛)
- (高血压, causes, 头痛)
- (头痛, symptom_of, 感冒)

---

## 2. 为什么需要知识图谱

### 2.1 传统 RAG 的局限性

传统的 RAG（检索增强生成）只使用向量检索：

```
用户问题 --> Embedding --> 向量相似度搜索 --> 返回文档 --> LLM 生成
```

**问题**：向量检索只能找到与问题**语义相似**的文档，但无法进行**推理**。

**举例**：
- 问题："高血压会导致什么症状？"
- 文档A："高血压是一种常见的心血管疾病。"
- 文档B："头痛可能是感冒的症状。"
- 文档C："高血压可能导致头痛。"

向量检索可能只返回文档A（与"高血压"语义最相似），
但无法推理出"高血压 → 导致 → 头痛 → 症状属于 → 感冒"这条链路。

### 2.2 知识图谱的优势

知识图谱可以进行**多跳推理**：

```
问题：高血压会导致什么症状？

图谱遍历：
高血压 --(causes)--> 头痛
高血压 --(causes)--> 胸闷
高血压 --(causes)--> 头晕

回答：高血压可能导致头痛、胸闷、头晕等症状。
```

### 2.3 三种召回方式对比

| 特性 | 向量召回 | BM25 召回 | 图谱召回 |
|------|----------|-----------|----------|
| **匹配方式** | 语义相似度 | 关键词匹配 | 图谱结构遍历 |
| **擅长场景** | 同义词、上下文 | 精确术语 | 多跳推理 |
| **举例** | "心脏"匹配"心脏病" | "阿司匹林"精确匹配 | 高血压→头痛→感冒 |
| **响应速度** | 快（<100ms） | 中（100-500ms） | 快（<100ms） |
| **局限性** | 无法推理 | 无法理解同义词 | 需要预先构建图谱 |

**最佳实践**：三路召回结合使用，通过 RRF 融合算法取长补短。

---

## 3. 知识图谱 vs 向量检索

### 3.1 数据结构对比

**向量检索**：
```
文档 --> Embedding --> 向量 [0.1, 0.3, -0.2, ...]
                         |
                         v
                    存储在 Milvus 中
                         |
                         v
                    余弦相似度搜索
```

**知识图谱**：
```
实体 --(关系)--> 实体 --(关系)--> 实体
  |                |                |
  v                v                v
属性             属性             属性
```

### 3.2 查询方式对比

**向量检索**：
```python
# 将问题编码为向量
query_vector = embedding.encode("高血压的症状")

# 在 Milvus 中搜索相似向量
results = milvus.search(query_vector, top_k=5)
```

**知识图谱**：
```python
# 从问题中提取实体
entities = extract_entities("高血压的症状")  # -> [高血压]

# 在图谱中遍历
for entity in entities:
    neighbors = graph.query_neighbors(entity, max_hops=2)
    # -> [头痛, 胸闷, 头晕, ...]
```

### 3.3 适用场景

| 场景 | 推荐方式 | 原因 |
|------|----------|------|
| 通用问答 | 向量召回 | 语义理解能力强 |
| 术语查询 | BM25 召回 | 关键词匹配精确 |
| 因果推理 | 图谱召回 | 支持多跳遍历 |
| 综合问答 | 三路融合 | 取长补短 |

---

## 4. 项目架构设计

### 4.1 整体架构

```
milvus-fastapi/
├── api/
│   ├── routes.py          # 文档管理路由
│   ├── qa_routes.py       # RAG 问答路由
│   └── graph_routes.py    # 知识图谱路由（新增）
├── core/
│   └── config.py          # 配置管理（新增图谱配置）
├── db/
│   ├── milvus_client.py   # Milvus 连接管理
│   └── graph_store.py     # 图存储实现（新增）
├── schemas/
│   ├── document.py        # 文档数据模型
│   ├── qa.py              # 问答数据模型
│   └── graph.py           # 图谱数据模型（新增）
├── services/
│   ├── vector_service.py  # 文档向量服务
│   ├── rag_service.py     # RAG 问答服务
│   ├── session_service.py # 会话管理
│   └── graph_service.py   # 知识图谱服务（新增）
└── utils/
    ├── ollama_embedding.py # Embedding 模型
    ├── ollama_chat.py      # LLM 对话客户端
    ├── bm25_retriever.py   # BM25 召回
    ├── reranker.py         # Cross-Encoder 精排
    ├── entity_extractor.py # 实体抽取（新增）
    ├── relation_extractor.py # 关系抽取（新增）
    └── graph_retriever.py  # 图谱召回（新增）
```

### 4.2 数据流

**文档入库流程**：
```
文件上传
    |
    v
文本提取（PDF/DOCX）
    |
    v
文本清洗 + 分片
    |
    v
Embedding --> Milvus 存储
    |
    v
实体抽取 --> 关系抽取 --> 图谱存储  <-- 新增
```

**RAG 问答流程**：
```
用户问题
    |
    +------+------+------+
    |      |      |      |
    v      v      v      v
 Dense  Sparse   Graph   <-- 三路召回
    |      |      |
    +------+------+
           |
           v
      RRF 融合排序
           |
           v
   Cross-Encoder 精排（可选）
           |
           v
      LLM 生成回答
```

### 4.3 配置驱动设计

所有图谱功能都通过配置开关控制，不影响现有功能：

```env
# 知识图谱开关（默认关闭）
ENABLE_KNOWLEDGE_GRAPH=false

# 图存储后端（networkx 或 neo4j）
GRAPH_STORE_BACKEND=networkx

# 图谱召回权重（0-1）
GRAPH_RECALL_WEIGHT=0.2

# 最大遍历跳数
GRAPH_MAX_HOPS=2
```

---

## 5. 核心概念详解

### 5.1 实体（Entity）

实体是知识图谱中的节点，代表具体的医疗概念。

**实体类型**：
| 类型 | 说明 | 示例 |
|------|------|------|
| Disease | 疾病 | 高血压、糖尿病、冠心病 |
| Symptom | 症状 | 头痛、发热、胸闷 |
| Drug | 药物 | 阿司匹林、青霉素、胰岛素 |
| Procedure | 手术/操作 | 冠状动脉搭桥术、阑尾切除术 |
| Department | 科室 | 心内科、神经外科、急诊科 |
| AnatomicalPart | 解剖部位 | 心脏、肝脏、大脑 |
| MedicalDevice | 医疗器械 | 呼吸机、心电监护仪 |
| Other | 其他 | 不属于以上类型 |

**实体 ID 生成规则**：
```
{doc_id}::entity::{name}::{type}

示例：
doc1::entity::阿司匹林::Drug
doc1::entity::高血压::Disease
```

### 5.2 关系（Relation）

关系连接两个实体，表示它们之间的语义联系。

**关系类型**：
| 类型 | 说明 | 示例 |
|------|------|------|
| treats | 治疗 | [阿司匹林] --(treats)--> [头痛] |
| causes | 导致 | [高血压] --(causes)--> [头痛] |
| symptom_of | 症状属于 | [头痛] --(symptom_of)--> [感冒] |
| used_for | 用于 | [心电监护仪] --(used_for)--> [心脏手术] |
| belongs_to | 属于 | [心内科] --(belongs_to)--> [心血管中心] |
| part_of | 是...的一部分 | [左心室] --(part_of)--> [心脏] |
| interacts_with | 相互作用 | [华法林] --(interacts_with)--> [阿司匹林] |
| contradicts | 禁忌/矛盾 | [阿司匹林] --(contradicts)--> [出血性疾病] |

### 5.3 多跳遍历

多跳遍历是知识图谱的核心能力，可以沿着关系边访问多层邻居。

**单跳遍历**（max_hops=1）：
```
起始：阿司匹林

遍历出边：
阿司匹林 --(treats)--> 头痛
阿司匹林 --(treats)--> 发热
阿司匹林 --(causes)--> 胃出血

结果：3个实体 + 3个关系
```

**两跳遍历**（max_hops=2）：
```
起始：阿司匹林

第1跳：
阿司匹林 --(treats)--> 头痛
阿司匹林 --(treats)--> 发热

第2跳：
头痛 --(symptom_of)--> 感冒
发热 --(symptom_of)--> 流感

结果：5个实体 + 4个关系
```

---

## 6. 代码实现详解

### 6.1 数据模型（schemas/graph.py）

```python
class Entity(BaseModel):
    """知识图谱中的实体节点"""
    id: str           # 实体唯一标识
    name: str         # 实体名称，如"阿司匹林"
    type: str         # 实体类型，如"Drug"
    attributes: dict  # 扩展属性
    doc_id: str       # 来源文档 ID
    chunk_id: str     # 来源分片 ID

class Relation(BaseModel):
    """知识图谱中的关系边"""
    source_id: str    # 源实体 ID
    target_id: str    # 目标实体 ID
    relation_type: str # 关系类型
    confidence: float # 置信度（0-1）
    doc_id: str       # 来源文档 ID
```

### 6.2 图存储（db/graph_store.py）

采用**策略模式**设计，支持两种后端：

```python
class GraphStoreProtocol(Protocol):
    """图存储协议"""
    def add_entities(self, entities: list[Entity]) -> int: ...
    def add_relations(self, relations: list[Relation]) -> int: ...
    def query_entity(self, name: str) -> list[Entity]: ...
    def query_neighbors(self, entity_id: str, max_hops: int) -> tuple: ...
    def delete_by_doc(self, doc_id: str) -> tuple: ...

class NetworkXGraphStore:
    """基于 NetworkX 的内存图存储"""
    # 适合开发测试，零依赖
    
class Neo4jGraphStore:
    """基于 Neo4j 的生产级图存储"""
    # 适合生产环境，高性能
```

### 6.3 实体抽取（utils/entity_extractor.py）

使用 LLM 进行实体抽取：

```python
class EntityExtractor:
    def extract(self, text: str, doc_id: str) -> list[Entity]:
        # 1. 构造提示词
        prompt = f"从以下文本中提取医疗实体：{text}"
        
        # 2. 调用 LLM
        response = self._llm.chat(prompt)
        
        # 3. 解析 JSON 响应
        entities = json.loads(response)
        
        # 4. 去重并返回
        return deduplicate(entities)
```

### 6.4 关系抽取（utils/relation_extractor.py）

给定实体列表，抽取它们之间的关系：

```python
class RelationExtractor:
    def extract(self, text: str, entities: list[Entity]) -> list[Relation]:
        # 1. 将实体列表格式化
        entities_text = "\n".join(f"- {e.name}（{e.type}）" for e in entities)
        
        # 2. 构造提示词
        prompt = f"已知实体：{entities_text}\n从文本中抽取关系：{text}"
        
        # 3. 调用 LLM
        response = self._llm.chat(prompt)
        
        # 4. 解析并返回
        return parse_relations(response, entities)
```

### 6.5 图谱召回（utils/graph_retriever.py）

将图谱集成到 RAG 流水线：

```python
class GraphRetriever:
    def retrieve(self, question: str, top_k: int) -> list[dict]:
        # 1. 从问题中提取实体
        entities = self._entity_extractor.extract(question)
        
        # 2. 在图谱中查询
        for entity in entities:
            neighbors = self._graph.query_neighbors(entity, max_hops=2)
        
        # 3. 收集关联的分片 ID
        chunk_ids = collect_chunk_ids(neighbors)
        
        # 4. 从 Milvus 获取分片文本
        chunks = self._milvus.query(chunk_ids)
        
        return chunks
```

### 6.6 3-way RRF 融合（services/rag_service.py）

将三路召回结果融合排序：

```python
def _rrf_fusion(self, dense, sparse, graph, top_k):
    """
    RRF 公式：
    score(d) = w_dense / (k + rank_dense(d))
             + w_sparse / (k + rank_sparse(d))
             + w_graph / (k + rank_graph(d))
    """
    k = 60  # 常数，避免排名第一的文档权重过大
    
    for doc_id in all_ids:
        d_rank = dense_rank.get(doc_id, default_rank)
        s_rank = sparse_rank.get(doc_id, default_rank)
        g_rank = graph_rank.get(doc_id, default_rank)
        
        score = (
            w_dense / (k + d_rank)
            + w_sparse / (k + s_rank)
            + w_graph / (k + g_rank)
        )
```

---

## 7. GraphRAG 原理

### 7.1 什么是 GraphRAG

GraphRAG = Graph + RAG，即在传统 RAG 基础上引入知识图谱。

**传统 RAG**：
```
问题 --> 向量检索 --> 文档片段 --> LLM 生成
```

**GraphRAG**：
```
问题 --> 向量检索 --> 文档片段
   |                        |
   +-----> 图谱查询 ------->+
                              |
                              v
                         RRF 融合
                              |
                              v
                         LLM 生成
```

### 7.2 GraphRAG 的优势

1. **多跳推理**：可以推理"A导致B，B是C的症状"这种链式关系
2. **结构化知识**：图谱中的实体和关系是结构化的，比纯文本更精确
3. **补充召回**：可以找到向量召回遗漏的文档
4. **可解释性**：可以展示推理路径，增强回答的可信度

### 7.3 实现细节

**图谱上下文注入**：
```python
# 将图谱信息格式化为文本，注入 LLM 提示词
context = """
相关知识图谱信息：
[阿司匹林] --(treats)--> [头痛]
[高血压] --(causes)--> [头痛]
[头痛] --(symptom_of)--> [感冒]
"""

# LLM 可以利用这些结构化信息来回答问题
```

---

## 8. 如何使用

### 8.1 启用知识图谱

在 `.env` 文件中添加：

```env
# 启用知识图谱
ENABLE_KNOWLEDGE_GRAPH=true

# 使用 NetworkX 后端（零依赖）
GRAPH_STORE_BACKEND=networkx

# 图谱数据持久化路径
GRAPH_PERSIST_PATH=./data/graph.json
```

### 8.2 重启服务

```bash
uv run uvicorn main:app --reload
```

### 8.3 上传文档

```bash
# 上传文档（会自动构建图谱）
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@medical_doc.pdf"
```

### 8.4 查询图谱

```bash
# 查询图谱
curl -X POST "http://localhost:8000/api/v1/graph/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "高血压会导致什么症状？", "max_hops": 2}'
```

### 8.5 获取统计信息

```bash
curl "http://localhost:8000/api/v1/graph/stats"
```

### 8.6 可视化子图

```bash
# 获取以"高血压"为中心的子图
curl -X POST "http://localhost:8000/api/v1/graph/subgraph" \
  -H "Content-Type: application/json" \
  -d '{"entity_name": "高血压", "depth": 2}'
```

### 8.7 RAG 问答

```bash
# 问答（自动使用图谱召回）
curl -X POST "http://localhost:8000/api/v1/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "高血压会导致什么症状？"}'
```

---

## 9. 简历写法建议

### 9.1 项目描述

**项目名称**：基于 GraphRAG 的医疗知识问答系统

**技术栈**：Python / FastAPI / Milvus / NetworkX / Neo4j / Ollama

**项目描述**：
- 设计并实现了 GraphRAG 架构，将知识图谱检索与传统向量检索、BM25 检索相结合
- 使用 LLM 进行医疗领域实体和关系抽取，支持 8 种实体类型和 8 种关系类型
- 实现多跳推理能力，支持 2+ 跳图谱遍历，解决复杂因果链路问题
- 采用 3-way RRF 融合算法，将三路召回结果统一排序
- 支持可插拔图存储后端：NetworkX（开发）和 Neo4j（生产）
- 配置驱动设计，图谱功能完全可选，不影响现有 RAG 功能

### 9.2 技术亮点

1. **GraphRAG 架构**：知识图谱 + 向量检索 + BM25 三路融合
2. **LLM 实体/关系抽取**：零样本医疗领域信息抽取
3. **多跳推理**：2+ 跳图谱遍历，支持链式推理
4. **可插拔后端**：NetworkX / Neo4j 可配置切换
5. **配置驱动 + 优雅降级**：图谱功能完全可选
6. **医疗领域知识图谱**：疾病、症状、药物、手术、科室关系建模

### 9.3 面试常见问题

**Q：为什么选择 LLM 做实体抽取而不是 spaCy/HanLP？**
A：
1. 零样本能力：不需要标注训练数据
2. 中文支持好：大模型对中文理解能力强
3. 灵活性：通过修改 Prompt 可以调整抽取策略
4. 零依赖：复用现有 LLM 客户端

**Q：如何处理 LLM 抽取的不确定性？**
A：
1. 低温度生成（temperature=0.1）减少随机性
2. 实体去重：基于 (name, type) 去重
3. 类型校验：验证实体类型是否在预定义列表中
4. 异常容错：抽取失败时返回空列表

**Q：3-way RRF 融合的权重如何设置？**
A：
- 默认权重：dense=0.4, sparse=0.4, graph=0.2
- 可通过配置调整：GRAPH_RECALL_WEIGHT=0.2
- 建议：图谱权重不宜过大，因为图谱可能不完整

**Q：NetworkX 和 Neo4j 如何选择？**
A：
- NetworkX：零依赖，适合开发测试和小规模应用（<10万节点）
- Neo4j：需要 Docker，适合生产环境和大规模图谱（百万节点）
- 通过配置 GRAPH_STORE_BACKEND 切换

---

## 10. 进阶学习路线

### 10.1 基础知识

1. **图论基础**：有向图、无向图、BFS、DFS
2. **知识表示**：RDF、OWL、属性图
3. **NLP 基础**：NER（命名实体识别）、关系抽取

### 10.2 技术栈

1. **NetworkX**：Python 图计算库，本项目使用
2. **Neo4j**：图数据库，生产环境推荐
3. **spaCy/HanLP**：NLP 工具库，可选的实体抽取方案
4. **LangChain**：LLM 应用框架，有 GraphRAG 模块

### 10.3 进阶主题

1. **图神经网络（GNN）**：用深度学习处理图数据
2. **图嵌入（Graph Embedding）**：将图节点编码为向量
3. **图谱补全**：预测缺失的实体和关系
4. **动态图谱**：支持实时更新的知识图谱

### 10.4 推荐资源

1. **书籍**：
   - 《知识图谱：方法、实践与应用》
   - 《Graph Databases》（Neo4j 官方书籍）

2. **在线课程**：
   - Coursera: Knowledge Graphs
   - Stanford CS224W: Machine Learning with Graphs

3. **开源项目**：
   - Microsoft GraphRAG
   - Neo4j GenAI 生态
   - LangChain Graph RAG

4. **论文**：
   - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
   - "Graph Retrieval-Augmented Generation: A Survey"

---

## 附录：常见问题

### Q1：图谱构建失败怎么办？

A：检查以下几点：
1. LLM 服务是否正常运行（Ollama/OpenAI）
2. 网络连接是否正常
3. 查看日志中的错误信息
4. 尝试手动构建：`POST /api/v1/graph/build`

### Q2：图谱数据存储在哪里？

A：
- NetworkX：存储在 `./data/graph.json` 文件中
- Neo4j：存储在 Neo4j 数据库中

### Q3：如何清理图谱数据？

A：
- 删除单个文档的图谱：`DELETE /api/v1/graph/{doc_id}`
- 清空所有图谱：删除 `./data/graph.json` 文件

### Q4：图谱召回的结果分数为什么是 0.5？

A：图谱召回的初始分数固定为 0.5，实际分数会在 RRF 融合时根据排名重新计算。

### Q5：如何调整图谱召回的权重？

A：在 `.env` 中设置 `GRAPH_RECALL_WEIGHT=0.2`（默认 0.2，范围 0-1）
