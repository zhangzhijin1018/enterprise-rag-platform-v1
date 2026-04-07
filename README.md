# Enterprise RAG Platform

面向企业知识库的 **Agentic RAG** 工程化骨架：接入、Redis 热点答案缓存、MySQL FAQ 快速检索、澄清判定、多路查询、父子分层切块、混合检索、Milvus 向量检索、重排、LangGraph 编排、带引用的生成、拒答策略、RAGAS 评测、Prometheus / OpenTelemetry、Docker 与 Kubernetes。

## 要求

- **Python 3.10+**（团队 Conda 示例：`conda activate tmf_project`）
- 可选：`OPENAI_API_KEY`（真实 LLM / RAGAS 评测）；未配置时进入可审计的离线应答模式

## 快速开始

```bash
cd enterprise-rag-platform
conda activate tmf_project
cp .env.example .env
# 建议在 conda 环境内安装，避免与用户目录 ~/.local 的 pip 包混用
python -m pip install -e ".[dev]"
mkdir -p data/vector_store data/milvus data/eval_reports
# 可选：写入模拟知识库（Markdown），便于本地联调 /chat
make seed-mock
make api
# 另开终端：curl http://127.0.0.1:8000/healthz
```

模拟文档位于 `data/mock_corpus/`；入库脚本为 `infra/scripts/seed_mock_index.py`（`make seed-mock`）。

## 交互前端（流程演示）

高端简约深色控制台（Vite + React + Tailwind），覆盖 **问答 / 入库+重建索引 / FAQ 导入 / RAGAS 评测 / API 连接**。

```bash
# 终端 1：后端
conda activate tmf_project
make api

# 终端 2：前端（代理到 127.0.0.1:8000）
make web-install   # 首次
make web-dev
# 浏览器打开 http://127.0.0.1:5173
```

仅后端托管 UI：在项目根执行 `make web-build` 后启动 API，访问 **http://127.0.0.1:8000/** 将重定向至 **/ui/**。Docker 镜像构建阶段会一并编译前端。

前端 **回答区与检索片段** 使用 Markdown 渲染（GFM 表格/任务列表等 + `rehype-sanitize` 净化）。

## 目录结构

```
enterprise-rag-platform/
├── apps/api/           # FastAPI：chat / ingest / eval / health / metrics
├── apps/worker/        # 异步任务（ingest job；队列 consumer 占位）
├── core/
│   ├── orchestration/  # LangGraph、state、nodes、policies
│   ├── retrieval/      # BM25、dense、hybrid、reranker、IndexStore
│   ├── ingestion/      # parsers / chunkers / cleaners / metadata
│   ├── generation/     # prompts、LLM、citation、context 格式化
│   ├── evaluation/     # RAGAS runner + 样例数据集
│   ├── observability/  # logging、metrics、tracing
│   └── services/       # RAGRuntime 依赖组装
├── infra/              # Docker、K8s、Prometheus、脚本
├── tests/              # unit / integration / eval / load(locust)
├── docs/               # 架构、API、评测、部署
├── settings.py         # 兼容入口，转发 core.config.settings
├── pyproject.toml
├── docker-compose.yml
└── Makefile
```

## 主要 API

| 端点 | 说明 |
|------|------|
| `POST /chat` | `question`, `conversation_id?`, `top_k`, `stream` |
| `POST /ingest` | 上传 PDF/DOCX/PPTX/HTML/Markdown/TXT/CSV |
| `POST /faq/import` | 导入 FAQ CSV 到 MySQL，并刷新 FAQ 检索索引 |
| `POST /reindex` | 重建稠密向量并同步 Milvus / reload 检索器 |
| `POST /eval` | 运行 RAGAS（需密钥） |
| `GET /healthz` | 健康检查 |
| `GET /metrics` | Prometheus |

详见 [docs/api.md](docs/api.md)。

当前入库已覆盖 7 类常见企业文档格式：

- `PDF`：规章、手册、论文、导出报表
- `DOCX`：SOP、制度、方案文档
- `PPTX`：培训课件、汇报材料、架构分享
- `HTML`：知识库网页、帮助中心页面
- `Markdown`：技术文档、FAQ、运行手册
- `TXT`：日志摘录、FAQ 草稿、告警说明
- `CSV`：错误码表、FAQ 导出、配置项清单

## 配置

复制 `.env.example` 为 `.env`。当前默认同时启用：

- `Redis`：缓存热点答案与查询改写
- `MySQL`：存储 FAQ 结构化问答
- `Milvus Lite`：执行向量召回

关键变量：

- `REDIS_URL=redis://localhost:6379/0`
- `MYSQL_URL=mysql+pymysql://rag:rag@127.0.0.1:3306/enterprise_rag`
- `VECTOR_BACKEND=milvus`
- `MILVUS_URI=./data/milvus/enterprise_rag.db`
- `MILVUS_COLLECTION_NAME=rag_chunks`
- `VECTOR_STORE_PATH=./data/vector_store`
- `FAQ_BM25_THRESHOLD=0.85`
- `ANSWER_CACHE_TTL_SEC=86400`
- `EMBEDDING_MODEL_NAME=BAAI/bge-m3`
- `RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3`

说明：

- `Redis -> MySQL FAQ -> RAG` 是当前问答链路的执行顺序。
- Redis 命中时直接返回热点答案。
- MySQL FAQ 命中且置信度达到阈值时直接返回 FAQ 答案。
- `Milvus` 负责 dense retrieval。
- 本地 `chunks.jsonl / embeddings.npy` 继续保留，作为 BM25、parent chunk 回扩和可读调试镜像。
- 如果你要切到远端 Milvus Standalone / Cluster，只需要把 `MILVUS_URI` 改成类似 `http://127.0.0.1:19530`。

## 验证

```bash
pip install -e ".[dev]"
pytest tests/unit tests/integration -q
# 压测（需先启动 API）：
# locust -f tests/load/locustfile.py --host http://127.0.0.1:8000
# RAGAS（需 OPENAI_API_KEY）：
# pytest tests/eval -q
```

## Docker

```bash
docker compose up --build
```

如果 Docker 构建阶段需要下载较大的 Python 依赖，建议先在 `.env` 中配置：

```bash
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
```

`docker-compose.yml` 会把这个变量透传给镜像构建阶段的 `pip install`。依赖层默认会复用 Docker 构建缓存；只有 `pyproject.toml`、`Dockerfile`、前端依赖文件或 `PIP_INDEX_URL` 发生变化，或者本机清理了 Docker build cache，才会重新下载依赖。

## 扩展方向

- 图检索、多租户字段、权限过滤、Milvus 过滤表达式增强、队列化 Worker、对话记忆存储。

更多设计说明见 [docs/architecture.md](docs/architecture.md)。
