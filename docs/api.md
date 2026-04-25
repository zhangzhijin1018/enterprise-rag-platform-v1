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
| POST | `/reindex` | 依据 Milvus 中现有 chunk 文本重建向量矩阵，并重载检索器 |
| POST | `/eval` | 运行 RAGAS 评测（需 `OPENAI_API_KEY`） |
| GET | `/healthz` | 健康检查 |
| GET | `/metrics` | Prometheus 指标 |

## POST /chat

Request body:

```json
{
  "question": "string",
  "conversation_id": "optional",
  "history_messages": [
    {"role": "user", "content": "上一轮问题"},
    {"role": "assistant", "content": "上一轮回答"}
  ],
  "top_k": 8,
  "stream": false,
  "user_id": "optional",
  "username": "optional",
  "department": "optional",
  "role": "optional",
  "project_ids": ["optional"],
  "clearance_level": "optional",
  "query_scene": "optional",
  "require_citations": true,
  "allow_external_llm": null,
  "session_metadata": {}
}
```

`top_k` 映射为 **rerank Top-N**；稀疏/稠密检索 `top_k` 在此基础上放大并受 `settings` 上限约束。当前 retrieval 阶段还会结合 `query_scene / preferred_retriever / top_k_profile` 再做一次动态缩放，因此前端只需要传稳定的 `top_k`，后端会按问题类型决定召回宽度。

Response：见 `apps/api/schemas/chat.py`（`citations` 为结构化列表）。

当前响应除基础 `answer / confidence / citations / retrieved_chunks` 外，还包括：

- `refusal`
- `refusal_reason`
- `answer_mode`
- `data_classification`
- `model_route`
- `analysis_confidence`
- `analysis_source`
- `analysis_reason`
- `conflict_detected`
- `conflict_summary`
- `trace_id`
- `audit_id`

同时，当前服务还会在响应头返回：

- `X-Trace-ID`

这个值和本地 `logs/app.log / logs/audit.log` 里的 `trace_id` 一致，便于按单次请求回放排障。
同时，响应体里现在也会返回 `trace_id`，便于前端直接展示和复制。

当前 `conflict_detected / conflict_summary` 的含义是：

- `conflict_detected = true`
  说明最终候选上下文里，命中了同主题但版本、生效日期或权威级别存在差异的证据
- `conflict_summary`
  说明当前检测到了哪类结构化冲突，以及系统优先采用了哪份证据

当前 `citations` 结构除了基础引用字段外，还会尽量补充企业知识治理相关字段：

- `doc_type`
- `owner_department`
- `data_classification`
- `version`
- `effective_date`
- `authority_level`
- `source_system`
- `business_domain`
- `process_stage`
- `section_path`
- `matched_routes`
- `retrieval_score / semantic_score / governance_rank_score`
- `selection_reason`

这几个字段的语义分别对应：

- `doc_type`：原始文档类型，如 `pdf / docx / markdown`
- `owner_department`：归属部门或责任部门
- `data_classification`：文档密级，如 `public / internal / sensitive / restricted`
- `version`：制度或手册版本
- `effective_date`：生效日期
- `authority_level`：权威级别，当前是 `high / medium / low`
- `source_system`：来源系统，如 `local_file / oa`
- `business_domain / process_stage`：文档所属业务域与流程阶段
- `section_path`：chunk 在文档结构中的章节路径
- `matched_routes`：该证据是被哪些 query route 命中的
- `selection_reason`：系统把该证据留在最终上下文中的简要解释

当前 `/chat` 的执行顺序：

1. 从请求体构造 `user_context`
2. 从 `user_context` 构造 `access_filters`
3. 由 request middleware 生成 `trace_id`
4. 生成 `audit_id`
5. 先做 request-level 风控评估
6. 在允许 fast path 的情况下，先查 Redis 热点答案缓存
7. 再查 MySQL FAQ 检索
8. 如果都没命中，再进入完整 RAG

补充：

- 当前 request-level 风控前，已经预留了 `ML risk hint` 注入位。
- 如果启用 `ENABLE_ML_RISK_HINT=true`，系统会先基于 `question + session_metadata + user_context` 生成 `risk_level_hint`，再交给现有 `RuleBasedRiskEngine` 做最终裁决。
- 当前第一轮只接 request-level，默认关闭；即使 `onnx` provider 依赖缺失或推理失败，也会回退到纯规则模式，不影响主链路继续执行。

当前完整 RAG 的内部顺序：

1. `analyze_signals`：抽取 `strategy_signals`
   当前会额外返回 `analysis_confidence / analysis_source / analysis_reason`
2. `clarify_gate`：判断是否先追问
3. `resolve_context`：在多轮追问场景下生成 `resolved_query`
4. `build_query_plan`：输出 `rewritten_query / keyword_queries / multi_queries / hyde_query / structured_filters`
5. `retrieve`：执行多路混合检索，并合并 `metadata_intent + structured_filters + access_filters`
6. `ACL fallback filter`：对 child hits 做访问控制与数据分级兜底过滤
7. `metadata boost`：按 business domain / doc number / structured filter 命中情况轻量提权
8. `enterprise entity boost`：对 `department / plant / system_name / business_domain` 这类企业实体归一信号追加更强的排序 bonus
9. `resolve_data_classification / resolve_model_route`
10. `retrieval-level risk evaluation`
11. `rerank`：先裁剪候选，再交给 cross-encoder
12. `generate-level risk evaluation`
13. `generate`：先做 context packing，再组 prompt
14. `validate`

说明：

- 当前 `/chat` 请求与响应结构已经扩展到企业问答语义，不再只是最小问答字段。
- `history_messages` 是可选兼容字段；前端愿意传时，可显著提升多轮承接问题的 `resolved_query` 效果。
- 当前仓库还没有默认启用持久化会话历史存储，因此 `resolve_context` 会在“有 history 输入时增强、无 history 时安全退化”。
- 如果请求带了企业安全上下文，而且系统启用了 `ACL / data classification / model routing`，当前会优先绕过尚未 ACL 化的 fast path。
- 当前 `model_route` 是**路由决策标签**，用于表达安全策略，不代表系统已经一定完成真实多模型执行切换。
- 当前统一风控接口已经接入 `/chat / retrieve / generate` 三个执行点，默认实现为 `RuleBasedRiskEngine`。
- 当前本地日志已经默认落盘为 `logs/app.log` 和 `logs/audit.log`，并且关键步骤会以 `INFO` 级别记录 `trace_id / audit_id / user_id / department / role / event` 等字段。
- 如果 request-level 风控判定为明显高风险批量导出请求，`/chat` 会在进入图执行前直接拒答。
- 当前 `access_denied` 与 `no_relevant_chunks` 会区分返回，分别表示“有候选但无权访问”和“知识库里没有足够相关证据”。
- 当前重排已经开始叠加企业治理优先级：`authority_level / effective_date / version` 会在语义重排之后进一步影响最终上下文顺序。
- 当前 retrieval 不再把所有 route 一视同仁处理，`precise / balanced / broad` 会影响 sparse / dense 的实际 top_k。
- 当前 retrieval 还会按 `preferred_retriever` 和 `top_k_profile` 裁掉低价值 route，避免 `precise` 场景下的 `sub_query / hyde` 继续放大检索成本。
- 当前检索编码后端已切成“`BGEM3` 优先”：如果环境里已安装 `pymilvus[model] + FlagEmbedding`，`SparseRetriever / DenseRetriever` 会优先使用 `BGEM3EmbeddingFunction` 的 sparse+dense 表示；如果环境暂时缺依赖，则会自动回退到 `BM25 + SentenceTransformer`，但接口和主链路行为保持一致。
- 当前默认模型路径也已切到项目内本地目录：`EMBEDDING_MODEL_NAME=./modes/bge-m3`、`RERANKER_MODEL_NAME=./modes/bge-reranker-large`。
- 当前 `analyze_query` 不再是单纯 regex 分流，而是“规则 + 置信度 + LLM 增强 + guardrail”混合方案；当分析结果不稳定时，会回退到保守 `hybrid` 检索，避免误路由。
- 当前 `/chat` 已会把这层 query understanding 信号直接返回给前端和调用方，便于排查“为什么系统把这题当成 broad / hybrid / sparse”。
- 当前 query understanding 的高频业务词已支持走外部词典配置，默认文件为 `data/config/query_understanding_vocab.json`；如果你扩新疆能源业务词，建议优先改词典而不是直接改 `analyze_query.py`。
- 词典维护建议见：[docs/query_understanding_vocab.md](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/docs/query_understanding_vocab.md)
- 当前词典除了业务域词，还支持 `department_aliases / site_aliases / system_aliases`，命中后会直接进入 `metadata_intent`，例如 `department / plant / applicable_site / system_name`。
- 当前默认词典已经补到一版新疆能源专属场景，覆盖燃料、发电、应急、合同等业务域，以及 `燃管部 / 一号输煤线 / 燃料平台` 这类内部表达。
- 当前运行时会把词典预编译成内存索引，请求侧直接复用已编译 regex 和 alias 索引，不会每次重新读完整词典文件。
- 当前这些企业实体归一结果已经不只进入 filter，还会在 fusion 后触发 `enterprise entity boost`，优先把“部门 / 场站 / 系统 / 业务域”更对口的证据往前推。
- 当前 retrieval filter 对 `department / owner_department`、`plant / applicable_site` 这类同义实体组，已经采用“组内 OR、组间 AND”的匹配语义；也就是说，文档只要命中组内任一规范字段，就不会因为企业别名归一导致被误过滤。
- 当前 rerank 之前已经新增候选裁剪，避免 cross-encoder 总是处理完整 `hybrid_top_k`。
- 当前生成前已经新增 `context packing`，会限制 prompt 里参与生成的文档数、单文档 chunk 数和总字符数。
- 当前冲突检测是保守型结构化检测，主要关注版本、生效日期、权威级别差异；它不是通用语义矛盾识别器。
- 当前生成前还会经过 `egress policy`：
  - `internal` 默认脱敏后再发外部模型
  - `sensitive` 默认只发脱敏后的最小必要上下文
  - `restricted` 默认禁止出域；如果启用了本地占位执行，则优先走本地受限模式，否则拒答
- 当前 API schema 的字段 description 已补成中文，便于直接从代码阅读请求/响应语义。
- 当前 citation formatter 的字段说明也已补成中文，后续如果你顺着 citation 看前后链路，会更容易读。
- 当前 query understanding / query planning 相关源码也已补中文说明；如果你要从接口一路追到 `strategy_signals -> query plan`，阅读成本会低很多。
- 当前生成侧的 `context_format / generate_answer / answer_builder` 也已补中文说明；如果你要继续往下追“prompt 里到底喂了什么、答案和 citation 是怎么解析出来的”，阅读成本会低很多。
- 当前风控与审计侧的 `risk_engine / audit / egress_policy` 也已补中文说明；如果你要继续往下追“为什么这题被拒答、为什么只能 local_only、日志里到底记了什么”，阅读成本会低很多。
- 当前 ingestion 侧的 `basic metadata extractor / semantic chunker / pipeline` 也已补中文说明；如果你要继续往上追“原始文档是怎么变成可检索 chunk 的”，阅读成本会低很多。
- 当前不同文件类型虽然仍走统一 `SemanticChunker`，但已经会按 `doc_type / mime_type` 走不同 chunk profile；例如 PDF 会优先按页切、CSV 会走更紧凑的行级参数。

非流式响应示例：

```json
{
  "answer": "mock answer",
  "confidence": 0.42,
  "fast_path_source": null,
  "citations": [],
  "retrieved_chunks": [],
  "refusal": false,
  "refusal_reason": null,
  "answer_mode": "grounded_answer",
  "data_classification": "internal",
  "model_route": "external_allowed",
  "conflict_detected": false,
  "conflict_summary": null,
  "audit_id": "3a4a2d0e2c1f4f7b9b2d7e6f3c2a1b0d"
}
```

其中单条 citation 的典型结构会类似：

```json
{
  "doc_id": "policy-001",
  "chunk_id": "policy-001:child:0001",
  "title": "设备巡检管理制度",
  "source": "inspection_policy.pdf",
  "page": 3,
  "section": "巡检频率",
  "doc_type": "pdf",
  "owner_department": "设备管理部",
  "data_classification": "internal",
  "version": "2.1",
  "effective_date": "2025-04-08",
  "authority_level": "high",
  "source_system": "oa"
}
```

流式 NDJSON 当前会输出三类事件：

1. `meta`
   提前返回 `retrieved_chunks / citations / refusal / model_route / audit_id`
2. `token`
   渐进返回模型输出文本
3. `final`
   返回结构化最终结果，字段语义尽量与非流式保持一致

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

- 文本、metadata 和 dense 向量当前统一以 Milvus collection 作为权威存储
- parent / child chunk 的回扩也优先从 Milvus 中按 `chunk_id / parent_chunk_id` 查询

当前入库阶段会优先补齐并透传这几类企业 metadata：

- 文档身份与治理字段：`doc_number / version / version_status / status / effective_date / expiry_date / doc_category / doc_type / authority_level / data_classification / source_system`
- 组织与责任字段：`group_company / subsidiary / plant / department / owner_department / issued_by / approved_by / owner_role`
- 业务语义字段：`business_domain / process_stage / applicable_region / applicable_site / equipment_type / equipment_id / system_name / project_name / project_phase`
- chunk 局部语义字段：`section_path / section_level / section_type / topic_keywords / chunk_summary`
- ACL 相关字段：`allowed_users / allowed_roles / allowed_departments / project_ids`

说明：

- 标量字段会被规范成稳定字符串，列表字段会被保留为列表，避免 `project_ids / allowed_departments` 在后续过滤里退化成模糊字符串匹配。
- `metadata_filters` 当前已支持列表型 metadata 匹配，因此这些字段后续既可参与检索前过滤，也可参与召回后的 ACL fallback。
- `doc_number / owner_department / doc_type / data_classification / effective_date / authority_level / business_domain / process_stage / equipment_type / project_name / section_type` 这些字段已经提升为 Milvus collection 一级 schema，后续 retrieval 可以优先服务端过滤。

## POST /eval

作用：

- 触发一轮离线评测
- 返回 `report_path`
- 返回 `analysis_path`
- 返回摘要 `summary`

其中：

- `report_path` 指向完整 JSON 评测报告
- `analysis_path` 指向 Markdown explainability report，适合直接做 badcase 回放

当前 `summary` 除了 RAGAS 指标外，还会带企业治理相关统计，例如：

- `sample_count`
- `refusal_rate`
- `conflict_detected_rate`
- `classification:internal`
- `model_route:external_allowed`
- `scenario:policy_conflict`
- `tag:acl`
- `expected_refusal_match_rate`
- `expected_conflict_match_rate`
- `matched_route:*`
- `metadata_boost_hit_rate`
- `governance_boost_hit_rate`
- `avg_explainable_citations`

详细逐题信息仍在落盘报告里查看，报告行级数据当前会保留：

- `answer / contexts`
- `refusal / refusal_reason`
- `answer_mode`
- `data_classification`
- `model_route`
- `conflict_detected / conflict_summary`
- `audit_id`

当前问答主链路也会产出统一审计事件，包含：

- `request_received`
- `prompt_audited`
- `output_audited`
- `response_sent`

审计事件当前会额外保留这些风控字段：

- `risk_level`
- `risk_action / risk_reason`

日志中默认保存脱敏后的 preview 与哈希，而不是原始明文 prompt/output。

当前还会按条件产出额外的 `security_alert` 事件，常见触发条件包括：

- 高风险 query
- `restricted` 数据
- `access_denied / restricted_data_local_only`
- `conflict_detected = true`
- 风控引擎显式要求告警

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

## POST /reindex

当前 `/reindex` 的语义已经更新为：

- 不重新解析原始文件
- 不依赖本地 `chunks.jsonl / embeddings.npy`
- 直接把 Milvus 中现有 chunk 文本作为唯一数据源，重新生成 dense 向量

这意味着：

- 如果只是 embedding 模型变了，优先执行 `/reindex`
- 如果 parser / metadata / chunk 切分规则变了，应该重新 `/ingest`
