# FastAPI Milvus Vector Service

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Milvus](https://img.shields.io/badge/Milvus-2.4+-orange.svg)](https://milvus.io/)

基于 FastAPI + Milvus 的生产级 RAG 问答服务，支持文档管理、语义检索和多轮对话。

## 功能特性

### 文档管理
- 📄 **批量写入** — 支持批量插入/更新文档，自动生成向量嵌入
- 📁 **文件上传** — 支持 PDF、DOCX 文件上传，自动提取文本并分片向量化
- 🔍 **语义搜索** — 基于余弦相似度的向量检索
- 🗑️ **CRUD 操作** — 完整的文档增删改查接口

### RAG 问答
- 💬 **非流式问答** — 一次性返回完整回答
- 🌊 **流式问答** — SSE 实时逐块返回，前端体验更好
- 🔄 **多轮对话** — 基于 Redis 的会话管理，支持上下文连续对话

### 检索增强
- 🎯 **混合召回** — 向量召回 + BM25 关键词召回，双路互补
- 📊 **RRF 粗排** — Reciprocal Rank Fusion 融合两路召回结果
- 🎖️ **Cross-Encoder 精排** — 使用预训练模型逐对打分，提升精度

### LLM 集成
- 🦙 **Ollama** — 支持本地 Ollama 模型（qwen2.5、llama3 等）
- 🤖 **OpenAI 兼容** — 支持 OpenAI、DeepSeek、硅基流动等 API

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
│   └── qa_routes.py         # RAG 问答路由
├── core/
│   └── config.py            # 配置管理（pydantic-settings）
├── db/
│   └── milvus_client.py     # Milvus 连接与集合管理
├── schemas/
│   ├── document.py          # 文档相关模型
│   └── qa.py                # 问答相关模型
├── services/
│   ├── vector_service.py    # 文档向量服务
│   ├── rag_service.py       # RAG 问答服务
│   └── session_service.py   # Redis 会话服务
├── utils/
│   ├── ollama_embedding.py  # Ollama 嵌入实现
│   ├── ollama_chat.py       # Ollama 对话客户端
│   ├── openai_chat.py       # OpenAI 兼容对话客户端
│   ├── bm25_retriever.py    # BM25 关键词检索
│   ├── reranker.py          # Cross-Encoder 精排
│   ├── text_chunker.py      # 文本分片
│   ├── text_cleaner.py      # 文本清洗
│   └── file_parser.py       # PDF/DOCX 文件解析
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
│  1. 召回（Recall）                        │
│  ├─ 稠密向量召回：语义匹配                  │
│  └─ 稀疏 BM25 召回：关键词匹配（可选）       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  2. 粗排（Coarse Ranking）                │
│  └─ RRF 融合两路召回结果                    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  3. 精排（Fine Ranking）—— 可选            │
│  └─ Cross-Encoder 逐对打分                 │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  4. 生成（Generation）                    │
│  └─ 将召回文档作为上下文，调用 LLM 生成回答   │
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
- **sentence-transformers** — Cross-Encoder 精排
- **Redis** — 多轮对话会话存储
- **jieba** — 中文分词（BM25 检索）
- **PyMuPDF** — PDF 文件解析
- **python-docx** — DOCX 文件解析

