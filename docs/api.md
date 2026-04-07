# HTTP API

Base URL: `http://localhost:8000`

前端（构建后）托管在 **`/ui/`**；访问根路径 **`/`** 会 302 到 `/ui/`。开发调试推荐使用 Vite：`apps/web` 内 `npm run dev`（代理 API）。

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | 问答（`stream: true` 返回 NDJSON 流） |
| POST | `/ingest` | 上传文档，异步入库 |
| POST | `/faq/import` | 导入 FAQ CSV 到 MySQL，并刷新 FAQ 检索索引 |
| GET | `/faq` | 获取 FAQ 列表 |
| PATCH | `/faq/{id}` | 启用或停用 FAQ |
| GET | `/jobs/{job_id}` | 任务状态 |
| POST | `/reindex` | 依据本地 chunk 镜像重建向量矩阵，同步 Milvus 并重载检索器 |
| POST | `/eval` | 运行 RAGAS 评测（需 `OPENAI_API_KEY`） |
| GET | `/healthz` | 健康检查 |
| GET | `/metrics` | Prometheus 指标 |

## POST /chat

Request body:

```json
{
  "question": "string",
  "conversation_id": "optional",
  "top_k": 8,
  "stream": false
}
```

`top_k` 映射为 **rerank Top-N**；稀疏/稠密检索 `top_k` 在此基础上放大并受 `settings` 上限约束。

Response：见 `apps/api/schemas/chat.py`（`citations` 为结构化列表）。

当前 `/chat` 的执行顺序：

1. 先查 Redis 热点答案缓存
2. 再查 MySQL FAQ 检索
3. 如果都没命中，再进入完整 RAG

## POST /ingest

`multipart/form-data`，字段 `file`。响应：`{ "job_id", "status": "accepted" }`。

当前默认支持的文件类型：

- `PDF`
- `DOCX`
- `PPTX`
- `HTML`
- `Markdown`
- `TXT`
- `CSV`

说明：

- `TXT` 适合日志、FAQ 草稿、规则说明。
- `CSV` 会在入库前转成“字段名: 字段值”的结构化文本，更利于检索。
- `PPTX` 会按 slide 提取标题与正文，保留页级结构，便于后续切块与引用。

入库后的存储行为：

- 文本与 metadata 会保存在本地 `chunks.jsonl`
- embedding 会保存在本地 `embeddings.npy`
- 如果 `VECTOR_BACKEND=milvus`，同一批向量会同步写入 Milvus collection

## POST /faq/import

`multipart/form-data`，字段 `file`。CSV 推荐表头：

- `question`
- `answer`
- `keywords`
- `category`

导入完成后，系统会：

1. upsert 到 MySQL `faq_entries`
2. 重新构建 FAQ 的 BM25 内存索引
3. 后续 `/chat` 可直接命中这批 FAQ

## GET /faq

返回当前 FAQ 全量列表，包含：

- `id`
- `question`
- `answer`
- `keywords`
- `category`
- `enabled`

这个接口主要给前端 FAQ 管理页使用。

## PATCH /faq/{id}

请求体：

```json
{
  "enabled": false
}
```

作用：

- 启用或停用 FAQ
- 更新后会立即刷新 FAQ 的内存 BM25 索引
