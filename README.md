# FastAPI Milvus Vector Service

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Milvus](https://img.shields.io/badge/Milvus-2.4+-orange.svg)](https://milvus.io/)

基于 FastAPI + Milvus 的生产级 RAG 问答服务，支持文档管理、语义检索、知识图谱和多轮对话。

## 功能特性

### 文档管理
- **批量写入** — 支持批量插入/更新文档，自动生成向量嵌入
- **文件上传** — 支持 PDF、DOCX 文件上传，自动提取文本并分片向量化
- **语义搜索** — 基于余弦相似度的向量检索
- **CRUD 操作** — 完整的文档增删改查接口

### RAG 问答
- **非流式问答** — 一次性返回完整回答
- **流式问答** — SSE 实时逐块返回，前端体验更好
- **多轮对话** — 基于 Redis 的会话管理，支持上下文连续对话

### 检索增强
- **Query 改写** — 查询扩展、HyDE、Step-back、关键词提取四种策略
- **混合召回** — 向量召回 + BM25 关键词召回，双路互补
- **RRF 粗排** — Reciprocal Rank Fusion 融合两路召回结果
- **Cross-Encoder 精排** — 使用预训练模型逐对打分，提升精度

### 知识图谱（GraphRAG）
- **自动构建** — 文档入库时自动抽取实体和关系，构建知识图谱
- **LLM 抽取** — 基于大模型的实体/关系抽取，支持 8 种医疗实体类型
- **多跳推理** — 支持 2+ 跳图谱遍历，实现复杂因果链路推理
- **3-way RRF** — 向量召回 + BM25 + 图谱召回三路融合
- **可视化支持** — 提供子图查询接口，支持前端图谱可视化
- **可插拔后端** — 支持 NetworkX（开发）和 Neo4j（生产）两种存储后端

### LLM 集成
- **Ollama** — 支持本地 Ollama 模型（qwen2.5、llama3 等）
- **OpenAI 兼容** — 支持 OpenAI、DeepSeek、硅基流动等 API

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/yourusername/fastapi-milvus-vector-service.git
cd fastapi-milvus-vector-service

# 安装依赖（使用 uv）
uv sync

# 配置环境
cp .env.example .env

# 启动开发服务器
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

启动后访问：
- Swagger 文档：<http://127.0.0.1:8000/docs>
- ReDoc 文档：<http://127.0.0.1:8000/redoc>

## 项目结构

```text
├── main.py                  # 应用入口与生命周期管理
├── api/
│   ├── routes.py            # 文档管理路由
│   ├── qa_routes.py         # RAG 问答路由
│   └── graph_routes.py      # 知识图谱管理路由
├── core/
│   └── config.py            # 配置管理（pydantic-settings）
├── db/
│   ├── milvus_client.py     # Milvus 连接与集合管理
│   └── graph_store.py       # 图存储实现（NetworkX/Neo4j）
├── schemas/
│   ├── document.py          # 文档相关模型
│   ├── qa.py                # 问答相关模型
│   └── graph.py             # 知识图谱相关模型
├── services/
│   ├── vector_service.py    # 文档向量服务
│   ├── rag_service.py       # RAG 问答服务（含 3-way RRF）
│   ├── session_service.py   # Redis 会话服务
│   └── graph_service.py     # 知识图谱业务服务
├── utils/
│   ├── ollama_embedding.py  # Ollama 嵌入实现
│   ├── ollama_chat.py       # Ollama 对话客户端
│   ├── openai_chat.py       # OpenAI 兼容对话客户端
│   ├── bm25_retriever.py    # BM25 关键词检索
│   ├── reranker.py          # Cross-Encoder 精排
│   ├── text_chunker.py      # 文本分片
│   ├── text_cleaner.py      # 文本清洗
│   ├── file_parser.py       # PDF/DOCX 文件解析
│   ├── entity_extractor.py  # LLM 实体抽取
│   ├── relation_extractor.py# LLM 关系抽取
│   ├── graph_retriever.py   # 图谱召回器
│   └── query_rewriter.py    # Query 改写器
├── docs/
│   └── knowledge_graph_tutorial.md  # 知识图谱学习文档
└── tests/
```

## API 接口

### 文档管理

| 方法 | 接口 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（含集合信息） |
| POST | `/api/v1/documents/upsert` | 批量插入/更新文档 |
| POST | `/api/v1/documents/upload` | 上传 PDF/DOCX 文件 |
| POST | `/api/v1/documents/search` | 语义搜索 |
| GET | `/api/v1/documents/{id}` | 根据 ID 获取文档 |
| DELETE | `/api/v1/documents/{id}` | 根据 ID 删除文档 |

### RAG 问答

| 方法 | 接口 | 说明 |
|------|------|------|
| POST | `/api/v1/qa/ask` | 非流式问答 |
| POST | `/api/v1/qa/ask/stream` | 流式问答（SSE） |

### 知识图谱

| 方法 | 接口 | 说明 |
|------|------|------|
| POST | `/api/v1/graph/build` | 手动构建图谱 |
| GET | `/api/v1/graph/stats` | 获取图谱统计信息 |
| POST | `/api/v1/graph/query` | 查询图谱（多跳遍历） |
| POST | `/api/v1/graph/subgraph` | 获取子图（可视化） |
| DELETE | `/api/v1/graph/{doc_id}` | 删除图谱数据 |

## 使用示例

### 插入文档

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/upsert \
  -H 'Content-Type: application/json' \
  -d '{
    "items": [
      {
        "id": "doc-001",
        "text": "Milvus 是一个面向 AI 场景的向量数据库。",
        "source": "manual",
        "tags": ["milvus", "vector-db"]
      }
    ]
  }'
```

### 上传文件

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload \
  -F "file=@document.pdf" \
  -F "source=upload" \
  -F "tags=pdf,文档"
```

### 语义搜索

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query_text": "向量数据库是什么",
    "top_k": 5
  }'
```

### RAG 问答

```bash
# 非流式
curl -X POST http://127.0.0.1:8000/api/v1/qa/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Milvus 有什么优势？",
    "top_k": 5
  }'

# 流式
curl -X POST http://127.0.0.1:8000/api/v1/qa/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Milvus 有什么优势？",
    "top_k": 5
  }'
```

### 多轮对话

```bash
# 第一轮：获取 session_id
curl -X POST http://127.0.0.1:8000/api/v1/qa/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "什么是向量数据库？"
  }'

# 第二轮：传入 session_id 继续对话
curl -X POST http://127.0.0.1:8000/api/v1/qa/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "它有哪些应用场景？",
    "session_id": "返回的session_id"
  }'
```

### 知识图谱

```bash
# 查询图谱（多跳推理）
curl -X POST http://127.0.0.1:8000/api/v1/graph/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "高血压会导致什么症状？",
    "max_hops": 2,
    "top_k": 10
  }'

# 获取图谱统计
curl http://127.0.0.1:8000/api/v1/graph/stats

# 获取子图（用于可视化）
curl -X POST http://127.0.0.1:8000/api/v1/graph/subgraph \
  -H 'Content-Type: application/json' \
  -d '{
    "entity_name": "高血压",
    "depth": 2
  }'

# 手动构建图谱
curl -X POST http://127.0.0.1:8000/api/v1/graph/build \
  -H 'Content-Type: application/json' \
  -d '{
    "doc_id": "doc-001",
    "text": "患者有高血压病史，长期服用阿司匹林100mg。"
  }'
```

## 配置说明

通过 `.env` 文件配置：

```env
# 应用配置
APP_NAME=Milvus FastAPI
APP_ENV=dev

# Milvus 配置
# 本地 Milvus Lite（默认，无需独立服务）
MILVUS_URI=./data/milvus_demo.db
# 或远程 Milvus/Zilliz Cloud
# MILVUS_URI=http://localhost:19530
# MILVUS_TOKEN=root:Milvus

# Embedding 配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# 文本分片
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# LLM 配置
LLM_PROVIDER=ollama  # 或 openai
OLLAMA_CHAT_MODEL=qwen2.5:7b
# OPENAI_API_KEY=sk-xxx
# OPENAI_BASE_URL=https://api.openai.com/v1

# 混合召回（可选）
ENABLE_HYBRID_RECALL=false
HYBRID_RECALL_ALPHA=0.5

# Cross-Encoder 精排（可选）
ENABLE_RERANKER=false
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Query 改写（可选）
ENABLE_QUERY_REWRITE=false
QUERY_REWRITE_STRATEGY=all  # expansion/hyde/stepback/keywords/all
QUERY_EXPANSION_COUNT=3

# 知识图谱（可选）
ENABLE_KNOWLEDGE_GRAPH=false
GRAPH_STORE_BACKEND=networkx  # 或 neo4j
GRAPH_PERSIST_PATH=./data/graph.json
GRAPH_MAX_HOPS=2
GRAPH_RECALL_WEIGHT=0.2

# Neo4j 配置（仅 GRAPH_STORE_BACKEND=neo4j 时需要）
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=your_password

# Redis 会话存储（可选）
REDIS_URL=redis://localhost:6379/0
SESSION_TTL_SECONDS=3600
```

## RAG 流水线

```text
用户问题
    │
    ▼
┌─────────────────────────────────────────┐
│  0. Query 改写（可选）                   │
│  ├─ 查询扩展：生成多个子问题              │
│  ├─ HyDE：生成假设性答案                 │
│  ├─ Step-back：生成更抽象的问题          │
│  └─ 关键词提取：提取检索关键词            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  1. 召回（Recall）                       │
│  ├─ 稠密向量召回：语义匹配                │
│  ├─ 稀疏 BM25 召回：关键词匹配（可选）     │
│  └─ 图谱召回：知识图谱多跳遍历（可选）     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  2. 粗排（Coarse Ranking）               │
│  └─ 3-way RRF 融合三路召回结果            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  3. 精排（Fine Ranking）—— 可选          │
│  └─ Cross-Encoder 逐对打分               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  4. 生成（Generation）                   │
│  └─ 将召回文档作为上下文，调用 LLM 生成回答│
└─────────────────────────────────────────┘
    │
    ▼
  回答（含引用标注 [1][2][3]）
```

## 技术栈

- **FastAPI** — 现代异步 Python Web 框架
- **Milvus** — 开源向量数据库，专为 AI 设计
- **Pydantic** — 数据验证与配置管理
- **Ollama** — 本地 LLM/嵌入模型服务
- **NetworkX** — Python 图计算库（知识图谱存储）
- **Neo4j** — 图数据库（可选，生产环境推荐）
- **sentence-transformers** — Cross-Encoder 精排
- **Redis** — 多轮对话会话存储
- **jieba** — 中文分词（BM25 检索）
- **PyMuPDF** — PDF 文件解析
- **python-docx** — DOCX 文件解析

## 文档

- [知识图谱学习指南](docs/knowledge_graph_tutorial.md) — GraphRAG 架构、实现原理、使用方法
