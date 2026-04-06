# HTTP API

Base URL: `http://localhost:8000`

前端（构建后）托管在 **`/ui/`**；访问根路径 **`/`** 会 302 到 `/ui/`。开发调试推荐使用 Vite：`apps/web` 内 `npm run dev`（代理 API）。

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | 问答（`stream: true` 返回 NDJSON 流） |
| POST | `/ingest` | 上传文档，异步入库 |
| GET | `/jobs/{job_id}` | 任务状态 |
| POST | `/reindex` | 依据 `chunks.jsonl` 重建向量矩阵并重载检索器 |
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

## POST /ingest

`multipart/form-data`，字段 `file`。响应：`{ "job_id", "status": "accepted" }`。
