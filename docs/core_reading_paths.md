# 主链路阅读清单

## 目的

这份文档不是罗列所有源码，而是把最关键的 5 条主链路拆开，按“文件顺序 + 阅读目标”组织成一个真正可执行的阅读清单。

适合两种场景：

1. 你要快速学会这个项目
2. 你要准备面试时按链路讲项目

---

## 阅读原则

每条链路都按同一顺序读：

1. 先看入口
2. 再看状态和编排
3. 再看核心执行节点
4. 最后看支撑层和数据结构

建议不要一天并行读太多条链路。  
最稳的顺序是：

1. 问答总入口
2. 查询理解链路
3. 检索链路
4. 生成链路
5. 风控链路
6. 入库链路

---

## 1. 问答总入口

### 适合什么时候看

- 第一次进入项目
- 想先建立全局感觉

### 阅读顺序

1. [apps/api/routes/chat.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/apps/api/routes/chat.py)
2. [apps/api/schemas/chat.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/apps/api/schemas/chat.py)
3. [core/services/runtime.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/services/runtime.py)
4. [core/orchestration/graph.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/graph.py)
5. [core/orchestration/state.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/state.py)

### 这一段要读懂什么

- `/chat` 如何进入系统
- 流式和非流式怎么分叉
- `user_context / access_filters / audit_id` 何时接入
- LangGraph 如何拿到初始 state

### 读完后你应该能复述

> 用户请求先进入 `/chat`，路由层构造企业上下文和审计信息，再把问题交给图编排层，图里再继续走查询理解、检索、生成和校验。

---

## 2. 查询理解链路

### 适合什么时候看

- 你想搞懂 route 怎么选
- 你想搞懂为什么某些问题会走 `keyword / sub_query / hyde`

### 阅读顺序

1. [core/orchestration/nodes/analyze_query.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/nodes/analyze_query.py)
2. [core/orchestration/query_understanding_vocab.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/query_understanding_vocab.py)
3. [data/config/query_understanding_vocab.json](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/data/config/query_understanding_vocab.json)
4. [core/orchestration/nodes/clarify_query.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/nodes/clarify_query.py)
5. [core/orchestration/nodes/resolve_context.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/nodes/resolve_context.py)
6. [core/orchestration/query_expansion.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/query_expansion.py)
7. [core/generation/prompts/templates.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/generation/prompts/templates.py)

### 这一段要读懂什么

- 规则层在抽什么 signal
- 什么时候触发 LLM 补判
- 为什么要有 guardrail 回退
- query vocab 如何把企业简称归一
- query plan 如何生成 rewrite / keyword / sub_query / hyde

### 读完后你应该能复述

> 这个项目不是直接让 LLM 决定检索路线，而是先用规则和词典抽信号，再按置信度决定是否需要 LLM 补判，最后生成多路 query plan。

---

## 3. 检索链路

### 适合什么时候看

- 你想搞懂为什么最终命中的是这些 chunk
- 你想搞懂 sparse / dense / hybrid / rerank / governance 怎么配合

### 阅读顺序

1. [core/orchestration/nodes/retrieve_docs.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/nodes/retrieve_docs.py)
2. [core/retrieval/metadata_filters.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/retrieval/metadata_filters.py)
3. [core/retrieval/access_control.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/retrieval/access_control.py)
4. [core/retrieval/sparse_retriever.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/retrieval/sparse_retriever.py)
5. [core/retrieval/dense_retriever.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/retrieval/dense_retriever.py)
6. [core/retrieval/milvus_retriever.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/retrieval/milvus_retriever.py)
7. [core/retrieval/hybrid_fusion.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/retrieval/hybrid_fusion.py)
8. [core/orchestration/nodes/rerank_docs.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/nodes/rerank_docs.py)
9. [core/retrieval/reranker.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/retrieval/reranker.py)
10. [core/retrieval/governance.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/retrieval/governance.py)
11. [core/retrieval/schemas.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/retrieval/schemas.py)

### 这一段要读懂什么

- `metadata_intent + structured_filters + access_filters` 怎么合并
- sparse 和 dense 分别负责什么
- Milvus 侧哪些字段能直推过滤
- hybrid fusion 如何合分数
- rerank 前为什么还要裁候选
- governance 排序为什么要在 semantic rerank 之后

### 读完后你应该能复述

> 检索不是简单“搜一下”，而是先按 query 信号决定 route，再按 metadata 和 ACL 做过滤，再做 sparse/dense 混合召回、rerank 和治理排序。

---

## 4. 生成链路

### 适合什么时候看

- 你想搞懂为什么回答有 citation
- 你想搞懂 prompt 里到底拼了什么

### 阅读顺序

1. [core/orchestration/nodes/generate_answer.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/nodes/generate_answer.py)
2. [core/generation/context_format.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/generation/context_format.py)
3. [core/generation/llm_client.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/generation/llm_client.py)
4. [core/generation/answer_builder.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/generation/answer_builder.py)
5. [core/generation/citation_formatter.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/generation/citation_formatter.py)
6. [core/orchestration/nodes/validate_grounding.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/nodes/validate_grounding.py)
7. [core/generation/local_executor.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/generation/local_executor.py)

### 这一段要读懂什么

- context packing 怎么限制 token 消耗
- 为什么只把部分 reranked hits 送进模型
- 模型输出为什么不能直接信任
- citation 为什么要做白名单校验
- local fallback 在什么场景触发

### 读完后你应该能复述

> 系统不是把检索结果直接丢给模型，而是先做 context packing，再按 grounded 格式调用模型，最后校验 chunk_id 和 citation，必要时走 local fallback。

---

## 5. 风控与审计链路

### 适合什么时候看

- 你想搞懂为什么有些问题被拒答
- 你想搞懂 `local_only / restricted / alert` 是怎么来的

### 阅读顺序

1. [core/security/risk_engine.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/security/risk_engine.py)
2. [core/generation/egress_policy.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/generation/egress_policy.py)
3. [core/observability/audit.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/observability/audit.py)
4. [core/retrieval/access_control.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/retrieval/access_control.py)
5. [apps/api/routes/chat.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/apps/api/routes/chat.py)

### 这一段要读懂什么

- `RiskContext` 如何构造
- `RiskDecision` 如何决定 allow / deny / local_only
- 出域策略为什么在生成前执行
- 审计事件里记录了什么，为什么不记录明文

### 读完后你应该能复述

> 企业级 RAG 的安全不是回答后再裁剪，而是请求进入、检索命中、生成出域三个阶段都要做风险判断和审计留痕。

---

## 6. 入库链路

### 适合什么时候看

- 你想搞懂文档进来后是怎么变成 chunk 的
- 你想搞懂为什么不同文件类型切块行为不一样

### 阅读顺序

1. [apps/api/routes/ingest.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/apps/api/routes/ingest.py)
2. [core/ingestion/pipeline.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/pipeline.py)
3. [core/ingestion/parsers/registry.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/parsers/registry.py)
4. [docs/ingestion_filetype_matrix.md](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/docs/ingestion_filetype_matrix.md)
5. [core/ingestion/parsers/pdf_parser.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/parsers/pdf_parser.py)
6. [core/ingestion/parsers/docx_parser.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/parsers/docx_parser.py)
7. [core/ingestion/parsers/pptx_parser.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/parsers/pptx_parser.py)
8. [core/ingestion/parsers/csv_parser.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/parsers/csv_parser.py)
9. [core/ingestion/chunkers/semantic_chunker.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/chunkers/semantic_chunker.py)
10. [core/ingestion/metadata_extractors/basic.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/metadata_extractors/basic.py)
11. [core/retrieval/index_store.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/retrieval/index_store.py)

### 这一段要读懂什么

- 不同 parser 先增强了什么结构
- 为什么是“统一 chunker + 文件类型 profile”
- parent / child chunk 的作用分工是什么
- metadata 何时写进 chunk

### 读完后你应该能复述

> 文档不是直接扔进 embedding，而是先 parser 增强结构、再做文件类型感知切块、再补 metadata，最后才进入索引层。

---

## 7. 评测与解释性链路

### 适合什么时候看

- 你想搞懂 badcase 是怎么沉淀的
- 你想搞懂 explainability report 从哪里来

### 阅读顺序

1. [apps/api/routes/eval.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/apps/api/routes/eval.py)
2. [apps/api/schemas/eval_schema.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/apps/api/schemas/eval_schema.py)
3. [core/evaluation/ragas_runner.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/evaluation/ragas_runner.py)
4. [docs/evaluation.md](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/docs/evaluation.md)
5. [core/evaluation/datasets/enterprise_eval.jsonl](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/evaluation/datasets/enterprise_eval.jsonl)

### 这一段要读懂什么

- `/eval` 如何触发离线评测
- JSON 报告和 Markdown explainability report 各自干什么
- 为什么现在不仅评估答案质量，还评估 route、boost、conflict、refusal

### 读完后你应该能复述

> 这个项目的评测已经不只是看 RAGAS 分数，而是把检索策略、拒答、冲突检测和 explainability 也纳入了闭环。

---

## 8. 最推荐的学习节奏

### 如果你只有 1 天

按这个顺序：

1. 问答总入口
2. 查询理解链路
3. 检索链路

### 如果你有 3 天

按这个顺序：

1. 问答总入口
2. 查询理解链路
3. 检索链路
4. 生成链路
5. 风控链路

### 如果你要准备面试

优先抓这 4 条：

1. 查询理解链路
2. 检索链路
3. 风控链路
4. 评测与解释性链路

---

## 9. 配套文档

建议和这份清单配合阅读：

- [docs/comment_coverage_audit.md](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/docs/comment_coverage_audit.md)
- [功能链路详解.md](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/功能链路详解.md)
- [核心技术原理.md](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/核心技术原理.md)
- [源码精读手册.md](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/源码精读手册.md)
- [docs/ingestion_filetype_matrix.md](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/docs/ingestion_filetype_matrix.md)
