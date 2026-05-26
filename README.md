# Milvus FastAPI

这是一个整理后的、扁平化目录结构的 `FastAPI + Milvus` 项目。

目标很明确：

- 所有核心代码直接放在项目根目录下，不再增加一层无意义的 `app/` 嵌套。
- 默认支持本地 `Milvus Lite`，开箱即跑。
- 同时保留连接远程 `Milvus / Zilliz Cloud` 的能力。
- 代码和文档都尽量详细注释，适合学习和继续扩展。

## 1. 项目结构

```text
milvus_fastapi/
├── .env.example
├── .gitignore
├── README.md
├── main.py
├── pyproject.toml
├── api/
│   └── routes.py
├── core/
│   └── config.py
├── db/
│   └── milvus_client.py
├── docs/
│   ├── framework_tutorial.md
│   └── implementation_notes.md
├── schemas/
│   └── document.py
├── services/
│   └── vector_service.py
├── tests/
│   └── test_demo_embedding.py
└── utils/
    └── demo_embedding.py
```

这套结构已经比较适合真实项目：

- `main.py`：应用入口与生命周期管理。
- `api/`：路由层，只处理 HTTP 协议。
- `services/`：业务层，负责写入、查询、删除等业务逻辑。
- `db/`：Milvus 客户端与集合初始化。
- `schemas/`：请求和响应模型。
- `core/`：配置层。
- `utils/`：通用组件，这里放示例 embedding。
- `docs/`：实现说明和框架教程。
- `tests/`：基础测试。

## 2. 为什么这样改

你要求：

- 当前工程目录名改成 `milvus_fastapi`
- 所有代码都直接写在这个文件夹下
- 不要做无意义的文件夹嵌套
- 清理多余文件

所以我把原来的模板子目录结构直接提升到了根目录，避免出现“根目录下面再套一层项目目录，再套一层 app”的过度嵌套。

## 3. 快速开始

```bash
cd /Users/nn/python_workspace/milvus_fastapi
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
cp .env.example .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

启动后访问：

- Swagger 文档：<http://127.0.0.1:8000/docs>
- ReDoc 文档：<http://127.0.0.1:8000/redoc>

## 4. 默认运行模式

默认配置：

```env
MILVUS_URI=./data/milvus_demo.db
```

这表示项目默认使用本地 `Milvus Lite`。

优点是：

- 不需要先手动启动独立 Milvus 服务。
- 拿到项目后本地就能直接跑通。
- 适合学习、调试和接口联调。

如果你要切换到远程 Milvus 或 Zilliz Cloud，可以把 `.env` 改成：

```env
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=root:Milvus
MILVUS_DB_NAME=default
```

## 5. API 示例

### 5.1 健康检查

```bash
curl http://127.0.0.1:8000/health
```

### 5.2 批量写入文档

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/upsert \
  -H 'Content-Type: application/json' \
  -d '{
    "items": [
      {
        "doc_id": "doc-001",
        "text": "Milvus 是一个面向 AI 场景的向量数据库。",
        "source": "manual",
        "tags": ["milvus", "vector-db"],
        "metadata": {"lang": "zh", "category": "database"}
      }
    ]
  }'
```

### 5.3 向量搜索

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query_text": "适合做向量检索接口的 Python 框架",
    "top_k": 3,
    "source": "manual"
  }'
```

### 5.4 获取单条文档

```bash
curl http://127.0.0.1:8000/api/v1/documents/doc-001
```

### 5.5 删除文档

```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/documents/doc-001
```

## 6. 文档说明

项目里有两份重点文档：

- `docs/implementation_notes.md`
  说明这个项目为什么这么设计。
- `docs/framework_tutorial.md`
  讲 FastAPI、项目分层、请求流程、Milvus 接入方式，适合你理解整个框架。

## 7. 后续扩展方向

如果你后面还想继续升级，我建议优先做这些：

- 接入真实 embedding 模型
- 增加 Docker Compose
- 增加多租户过滤字段
- 增加日志和鉴权
- 扩展成 RAG 检索服务
