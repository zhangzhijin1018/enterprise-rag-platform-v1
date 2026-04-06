# Architecture

## Layers

1. **Ingestion** — upload → parse (PDF/DOCX/HTML/Markdown) → clean → semantic chunk → persist chunks + embeddings (`IndexStore`).
2. **Retrieval** — BM25 (`SparseRetriever`), dense ANN over cosine similarity (`DenseRetriever`), fusion RRF/weighted (`HybridFusion`), cross-encoder rerank (`CrossEncoderReranker`).
3. **Query understanding** — lightweight classification + LLM query rewrite (Redis-cached); `core/orchestration/query_expansion.py`预留多查询扩展。
4. **Orchestration** — LangGraph `StateGraph`：`analyze → rewrite → retrieve → (empty|rerank) → generate → validate`；空召回走 `refuse_empty` 分支。
5. **Generation** — grounded prompt，强制 `[CHUNK_ID:…]` 与 `CITATIONS_JSON` 解析；离线模式无 API Key 时返回可审计占位答案。
6. **Evaluation** — RAGAS runner（faithfulness、answer relevancy、context recall/precision），报告写入 `data/eval_reports/`。
7. **Observability** — JSON 日志、Prometheus（`erp_*` 指标）、可选 OTLP gRPC 导出。
8. **Deployment** — Docker / docker-compose / Kubernetes 清单。

## Extension hooks

- 图检索：在 `retrieve_docs` 后插入子图或并行节点。
- 多租户：在 `ChunkMetadata.extra` 与索引过滤条件中扩展 `tenant_id`。
- 权限：在 `rerank` 前按 ACL 过滤 `RetrievedChunk`。
