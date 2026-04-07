# Architecture

## Layers

1. **Fast path** — Redis 热点答案缓存 → MySQL FAQ BM25 检索 → 命中则直接返回。
2. **Ingestion** — upload → parse (PDF/DOCX/PPTX/HTML/Markdown/TXT/CSV) → clean → parent-child semantic chunk → persist local chunk mirror (`IndexStore`) → sync vectors to Milvus.
3. **Retrieval** — child-only BM25 (`SparseRetriever`) over local chunk mirror, child-only dense ANN over Milvus (`MilvusDenseRetriever`), parent expansion, fusion RRF/weighted (`HybridFusion`), cross-encoder rerank (`CrossEncoderReranker`).
4. **Query understanding** — lightweight classification + clarification gate + query planning（Redis-cached）；支持 rewrite / multi-query / keyword query / HyDE。
5. **Orchestration** — LangGraph `StateGraph`：`analyze → clarify → rewrite → retrieve → (empty|rerank) → generate → validate`；需要澄清时走 `ask_clarify` 分支，空召回走 `refuse_empty` 分支。
6. **Generation** — grounded prompt，强制 `[CHUNK_ID:…]` 与 `CITATIONS_JSON` 解析；离线模式无 API Key 时返回可审计占位答案。
7. **Evaluation** — RAGAS runner（faithfulness、answer relevancy、context recall/precision），报告写入 `data/eval_reports/`。
8. **Observability** — JSON 日志、Prometheus（`erp_*` 指标）、可选 OTLP gRPC 导出。
9. **Deployment** — Docker / docker-compose / Kubernetes 清单（含 Redis / MySQL）。

## Extension hooks

- 图检索：在 `retrieve_docs` 后插入子图或并行节点。
- 多租户：在 `ChunkMetadata.extra` 与索引过滤条件中扩展 `tenant_id`。
- 权限：在 `rerank` 前按 ACL 过滤 `RetrievedChunk`。
