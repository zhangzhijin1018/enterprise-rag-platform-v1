# Architecture

## Layers

1. **Fast path** — Redis 热点答案缓存 → MySQL FAQ BM25 检索 → 命中则直接返回；但当前对带企业安全上下文的请求会优先绕过，避免 FAQ / cache 在未 ACL 化前造成权限绕过。
2. **Ingestion** — upload → parse (PDF/DOCX/PPTX/HTML/Markdown/TXT/CSV) → clean → enrich enterprise metadata → parent-child semantic chunk → encode with BGEM3-first `dense + sparse` pipeline → sync full snapshot to Milvus。当前 Milvus 已作为 chunk 原文、metadata、dense 向量、sparse 向量以及高频 metadata filter 字段标量索引的统一权威存储。
3. **Security context** — `ChatRequest` 在入口层被扩展为“问题 + 企业用户上下文”，当前 request middleware 会先生成 `trace_id`，随后问答入口再构造 `user_context / access_filters / audit_id`，并一路透传到完整图和 retrieval-only 链路。
4. **Risk engine** — 新增统一风控接口 `RiskContext / RiskDecision / RiskEngine`；当前默认实现为 `RuleBasedRiskEngine`，分别在 request、retrieval、generation 三个执行点做风险判断，并支持 `fail-open / fail-close`。当前 request-level 还新增了一个可选 `ML risk hint` provider，先产出 `risk_level_hint`，再由规则引擎完成最终裁决；默认关闭，失败时回退到纯规则模式。
5. **Retrieval** — child-only hybrid retrieval：默认由 `BGEM3EmbeddingFunction` 同时提供 dense + sparse 表示，并把两者一起写入 Milvus；在 `Milvus + BGEM3` 场景下，优先走 Milvus 原生 `hybrid_search`，并按 `query_scene` 映射到 `RRFRanker / WeightedRanker`；若 BGEM3 不可用，`SparseRetriever` 会降级到本地 BM25。检索阶段会合并 `metadata_intent + structured_filters + access_filters`，同义实体组采用“组内 OR、组间 AND”语义，并按 `query_scene / preferred_retriever / top_k_profile` 动态调整 sparse / dense `top_k`、执行 route pruning、child-hit ACL fallback、parent expansion、metadata boost、enterprise entity boost、cross-encoder rerank (`CrossEncoderReranker`)。
6. **Query understanding** — strategy signal extraction + heuristic confidence scoring + low-confidence LLM enhancement + conservative guardrail + clarification gate + context resolution + query planning（Redis-cached）；支持 `resolved / rewrite / multi-query / keyword / HyDE / structured_filters`。
7. **Orchestration** — LangGraph `StateGraph`：`analyze_signals → clarify_gate → resolve_context → build_query_plan → retrieve → (empty|rerank) → generate → validate`；需要澄清时走 `ask_clarify` 分支，空召回走 `refuse_empty` 分支，并保留 `access_denied` 等上游已确定的拒答语义。当前 `retrieve` 与 `generate` 节点都已接入统一风控决策。
8. **Generation** — grounded prompt，强制 `[CHUNK_ID:…]` 与 `CITATIONS_JSON` 解析；当前 citation formatter 已能把 `doc_type / owner_department / data_classification / version / effective_date / authority_level / source_system` 以及 `business_domain / process_stage / section_path / matched_routes / selection_reason` 暴露给 API / 前端；生成前还会经过 egress policy，根据数据分级决定“完整上下文 / 脱敏上下文 / 最小片段 / 本地占位执行 / 直接拒答”；随后新增 context packing，限制文档数、单文档 chunk 数和总字符数；离线模式无 API Key 时返回可审计占位答案。
9. **Model route & classification** — retrieval 后会基于命中结果收敛 `data_classification`，再结合请求策略与风控决策输出 `model_route` 决策标签，用于后续本地模型 / 外部模型 / 拒答策略分流。
10. **Evaluation** — RAGAS runner（faithfulness、answer relevancy、context recall/precision），报告写入 `data/eval_reports/`；当前还会把 `refusal / conflict_detected / data_classification / model_route / audit_id` 这些治理信号一并写入报告，并支持按 `scenario / tags / expected_refusal / expected_conflict` 做摘要统计。
11. **Observability** — JSON stdout 日志、Prometheus（`erp_*` 指标）、可选 OTLP gRPC 导出；当前还新增了本地 `logs/app.log / logs/audit.log` 两类 `.log` 文件。`app.log` 侧重排障，会以 `INFO` 级别记录 `trace_id / audit_id / user_id / department / role / event` 和关键步骤；`audit.log` 侧重审计与安全事件，记录请求、prompt、output 和响应阶段的脱敏审计信息，并可按高风险/高敏/冲突/权限拒答分流 `security_alert`。当前 `/chat` 响应头和响应体都会返回 `trace_id`，前端可直接展示，便于从页面定位到本地日志。
12. **Deployment** — Docker / docker-compose / Kubernetes 清单（含 Redis / MySQL）。

补充：

- 当前核心模型字段、RAGState 字段和 Milvus schema 字段都已经补了中文注释，后续阅读源码时可以直接结合代码看字段语义。
- 当前 settings、citation formatter、metadata filters 也已补中文说明，排查配置和检索过滤时不需要再反复猜字段语义。
- 当前 retrieval schemas、governance 和 retrieve_docs 也已补中文说明，顺着检索主链路读代码会更直接。
- 当前 query_understanding_vocab、analyze_query 和 query_expansion 也已补中文说明，顺着“query understanding -> query planning”链路读源码会更顺。
- 当前 context_format、generate_answer 和 answer_builder 也已补中文说明，顺着“上下文压缩 -> grounded generation -> citation 解析”链路读源码会更直接。
- 当前 risk_engine、audit 和 egress_policy 也已补中文说明，顺着“风控决策 -> 审计留痕 -> 最小必要出域”链路读源码会更直接。
- 当前 basic metadata extractor、semantic chunker 和 ingestion pipeline 也已补中文说明，顺着“文档解析 -> metadata 抽取 -> 分层切块 -> 入索引”链路读源码会更直接。
- 当前切块策略已经升级为“统一 chunker + 文件类型 profile”，不是每种格式一个全新 chunker，但会按 `doc_type / mime_type` 调整 PDF、PPTX、CSV、TXT 等格式的切块参数和 section 边界。
- 当前 parser 输出也做了文件类型增强：DOCX 会保留 heading level 和表格结构，PPTX 会保留 bullet 层级，CSV 会为每行补更强的主键标题，从而让统一 chunker 能吃到更多结构信号。
- 当前默认本地检索模型目录也已约定为：
  - `./modes/bge-m3`
  - `./modes/bge-reranker-large`
  这样 embedding 和 rerank 都可以直接走本地模型，不依赖首次在线下载。

## Extension hooks

- 图检索：在 `retrieve_docs` 后插入子图或并行节点。
- 多租户：在 `ChunkMetadata.extra` 与索引过滤条件中扩展 `tenant_id`。
- 权限：当前已在 `retrieve_docs` 的 child hits 阶段做 ACL fallback；后续可继续升级为更正式的 RBAC / ABAC / policy engine。
- 风控：当前默认是本地规则引擎；后续可替换为远程 PDP、OPA 或企业风控中心，而不改主链路执行点。
- fast path 安全：后续可给 FAQ / cache 层补 ACL，从“安全绕过”升级为“安全可用”。
- 模型路由：当前 `model_route` 仍是决策标签，后续可接入真实本地模型 / 外部模型执行器。

## Query routes

当前检索规划不再由固定业务分类驱动，而是按 query routes 组合执行：

- `original`：用户原始问题，默认保留，走 sparse + dense。
- `resolved`：多轮上下文补全后的完整问题，解决“这个 / 那个 / 今天谁值班 / 那如果是 3 号线呢”。
- `rewrite`：轻量检索改写后的主 query，默认偏保守，避免对精确实体和事实类问题过度改写。
- `sub_query`：复杂问题拆分出的多个检索视角，适合对比、原因+处理、方案类问题。
- `keyword`：适合 sparse 词面路线的关键词查询；当前优先走 BGEM3 sparse，降级时走 BM25。
- `hyde`：假设答案式 query，只走 dense，不默认总开。

## Strategy signals

`analyze_query` 当前输出的是 `strategy_signals`，而不是固定 `query_type`。典型信号包括：

- `need_history_resolution`
- `need_keyword_boost`
- `need_sub_queries`
- `need_hyde`
- `likely_structured_lookup`
- `has_precise_identifier`
- `has_time_constraint`
- `has_department_constraint`
- `has_person_constraint`
- `is_comparison`
- `is_multi_hop`
- `has_error_code`
- `query_scene`
- `preferred_retriever`
- `top_k_profile`
- `metadata_intent`
- `analysis_confidence`
- `analysis_source`
- `analysis_reason`

当前 `analyze_query` 的执行策略已经升级为：

1. 规则先抽 `strategy_signals`
2. 基于显式锚点、结构化约束和 query scene 计算 `analysis_confidence`
3. 当置信度低于阈值时，调用查询理解模型做结构化补判
4. 当最终置信度仍然过低时，回退到保守策略：
   - `preferred_retriever = hybrid`
   - `top_k_profile = balanced`
   - `need_hyde = false`

当前默认模型分层：

- 查询理解 / 澄清 / 多轮补全：`QUERY_UNDERSTANDING_MODEL_NAME=qwen-turbo`
- 查询规划：`QUERY_PLANNING_MODEL_NAME=qwen-turbo`
- 最终回答：`ANSWER_GENERATION_MODEL_NAME=qwen-plus`

当前规则层的高频业务词已经抽到词典配置：

- 默认路径：`./data/config/query_understanding_vocab.json`
- 维护说明见：[docs/query_understanding_vocab.md](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/docs/query_understanding_vocab.md)
- 配置项：`QUERY_UNDERSTANDING_VOCAB_PATH`

这层词典当前主要负责：

- 文档类别词：`policy / procedure / meeting / project`
- 设备词
- 部门后缀词
- 部门别名 / 场站别名 / 系统别名
- 结构化事实词
- follow-up 词
- 业务域词

当前默认词典已经补到一版新疆能源专属表达，包含：

- 燃料、发电、应急、合同管理等业务域词
- `燃料管理部 / 发电运行部 / 生产技术部 / 招标采购部` 等部门别名
- `一号输煤线 / 集控室 / 南露天煤场 / 翻车机卸煤区` 等场站别名
- `燃料管控平台 / 生产运营看板 / 设备缺陷管理系统` 等系统别名

当前词典运行时已经升级成“预编译内存索引”：

- 词典文件仍然是本地 JSON
- 但请求不会每次重新读文件、重新组 regex
- 运行时会缓存编译后的 scene pattern、部门 pattern、设备 pattern 和 alias 索引

这样 badcase 驱动调参时，优先改词典，不必先动核心路由代码。

## Clarify gate

`clarify_query` 已从“报错导向”升级为“缺槽位导向”，当前重点识别的槽位包括：

- `target_object`
- `department`
- `time_range`
- `person`
- `environment`
- `version`
- `comparison_targets`
- `symptom_description`
- `runtime_context`
- `shift`

## Structured filters

`structured_filters` 不是单独 route，而是附着在其他 route 上的约束条件。当前会从 query / resolved query 中提取：

- `time`
- `department`
- `shift`
- `person`
- `environment`
- `version`
- `line`

这些过滤条件会一路透传到 sparse / dense / Milvus dense 检索接口；当前本地检索器优先匹配 metadata / extra，Milvus 端会优先对 `department / shift / line / person / time / environment / version / doc_category` 等一级字段生成 filter expression，其余字段再走上层 post-filter 兼容。若检测到远端 collection 还是旧 schema，运行时会基于本地快照自动重建升级。

当前 retrieval 还会把 `metadata_intent` 作为一层“软过滤”与“轻量提权”信号：

- 没有显式结构化过滤时，先把 `metadata_intent` 作为默认检索约束下传
- 如果和显式 `structured_filters` 同名，则以显式过滤为准
- fusion 后按命中 `metadata_intent / structured_filters` 的程度追加小幅 score boost

这样做的目标不是替代 rerank，而是尽量让“更像企业真实答案”的候选更早排到前面，减少无效精排成本。

## Route pruning

当前 retrieval 不再默认把所有 route 都跑一遍，而是会基于 query profile 做轻量裁剪：

- `precise`
  - 默认裁掉 `hyde`
  - 不继续展开 `sub_query`
  - `keyword` route 也会限制数量
- `preferred_retriever = sparse`
  - `sub_query / keyword` 默认只走 sparse
- `preferred_retriever = dense`
  - `keyword` route 默认不再保留

这样做的核心目标是：先减少低价值检索调用，再谈后续 rerank 和 generation 优化。

## Access filters

`access_filters` 是当前企业化改造新增的第二类过滤条件，和 `structured_filters` 的职责不同：

- `structured_filters` 解决“找得准”
- `access_filters` 解决“看得到”

当前 `access_filters` 主要由 `/chat` 入口基于这些字段构造：

- `user_id`
- `department`
- `role`
- `project_ids`
- `clearance_level`

随后通过 `build_retrieval_acl_filters()` 映射到检索层更容易消费的 metadata 过滤字段，例如：

- `allowed_users`
- `allowed_departments`
- `allowed_roles`
- `project_ids`

## ACL fallback

当前版本除了把 access filter 下传给检索器，还会在 `retrieve_docs` 里对 child hits 再做一次本地 ACL / 数据分级兜底过滤。

这样做的原因：

1. 检索器 filter 更偏“候选缩小”，不是强安全引擎
2. 不同 backend 对 metadata 的支持能力不完全一致
3. 企业场景需要“下传过滤 + 本地兜底”双保险

当前 ACL fallback 的执行顺序是：

```text
多路 query route
-> sparse / dense search(filters = structured + access)
-> merge child hits
-> is_chunk_accessible()
-> expand to parent
-> fusion
```

## Data classification and model route

当前 retrieval 结束后会补两类企业安全语义：

1. `data_classification`
   当前按命中结果的最高等级收敛，等级顺序为：
   `public < internal < sensitive < restricted`
2. `model_route`
   当前是路由决策标签，典型值包括：
   - `default`
   - `external_allowed`
   - `local_preferred`
   - `local_only`

当前要特别注意：

- `model_route` 现在还不是“真实执行器切换”
- 它先用于表达系统安全决策
- 后续再接本地模型 / 外部模型执行层时，可以直接复用这套语义

## Governance ranking and conflict detection

这一轮开始，系统不再只依赖 cross-encoder 的语义分数做最终上下文排序，而是在其后叠加一层轻量企业治理优先级。

当前参与排序的字段包括：

- `authority_level`
- `effective_date`
- `version`

当前实现原则：

1. 先保留 cross-encoder 的语义相关性作为主排序基础
2. 再按配置化权重叠加 `governance_bonus`
3. 最终用 `governance_rank_score` 决定候选顺序
4. 不直接覆盖 `semantic_score`，避免后续阈值判断失真

这样做的好处是：

1. 不会因为“新版本”把明显不相关的文档硬顶上来
2. 能在相关候选之间优先选择“更新、更权威”的文档
3. trace 里可以明确解释为什么某份文档排在前面

当前冲突检测同样采取保守策略：

1. 只在最终候选上下文上做
2. 主要检查同主题证据里的 `version / effective_date / authority_level`
3. 如果发现差异，则返回 `conflict_detected / conflict_summary`
4. 生成阶段会把 `conflict_summary` 作为治理提示传给模型

要注意：

- 这不是通用语义矛盾检测
- 它当前更像“企业知识治理冲突提示”
- 目标是先把多版本、多权威来源显式暴露出来，而不是假装已经解决了所有文档冲突

## Retrieval performance notes

这一轮开始，检索性能优化不再只依赖更强 reranker，而是前移到了 retrieval planning：

1. `top_k_profile=precise`
   - 适合制度编号、错误码、结构化事实问题
   - 会主动收窄 sparse / dense 的候选数
2. `top_k_profile=balanced`
   - 当前默认档位
   - 兼顾召回与延迟
3. `top_k_profile=broad`
   - 适合对比、多文档综合、会议纪要追溯
   - 会主动放宽候选数

同时 `rerank_docs_node` 现在不会对所有 `fused_hits` 全量精排，而是先按：

- `rerank_candidate_multiplier`
- `rerank_candidate_max`

裁出更小的 rerank 输入集，再做 cross-encoder 精排。这一层直接减少了 rerank 延迟和模型成本。

## Context packing

生成前的上下文压缩现在单独前置到 `context_format.select_contexts_for_prompt()`：

1. 限制总文档数
2. 限制单文档 chunk 数
3. 限制总字符数
4. 同文档重复 `section_path` 默认只保留一份

它的目标不是“再做一次 rerank”，而是：

1. 避免同一文档的重复 chunk 挤占 prompt
2. 把 token 预算留给更多不同来源的证据
3. 降低大模型生成延迟与成本

## Citation explainability

这一轮开始，citation 不再只是“来源信息”，还会带出一部分最终排序解释：

- `matched_routes`
- `retrieval_score`
- `semantic_score`
- `governance_rank_score`
- `selection_reason`

设计原则是：

1. 解释信息优先复用已有 trace，不额外重复做一套排序系统
2. 解释保持短小，适合前端展示和面试演示
3. 不把内部实现细节全量暴露给 API，只保留最有价值的决策信号

这些 explainability 信号当前也已经进入评测链路：

- 每题会保留 `matched_routes / metadata_boosted / enterprise_entity_boosted / enterprise_entity_matches / governance_boosted / explainable_citation_count`
- `summary` 会额外统计 `matched_route:* / entity_match:* / metadata_boost_hit_rate / enterprise_entity_boost_hit_rate / governance_boost_hit_rate / avg_explainable_citations`

这样 badcase 分析时，就能同时看到“答得怎么样”和“为什么系统会这么选证据”。

## Enterprise metadata schema

当前企业化改造已经开始把“检索型 metadata”和“知识治理型 metadata”统一到入库阶段，而不是等到检索或生成时临时推断。

当前优先补齐两类字段：

1. 检索与结构化过滤字段
   - `department / shift / line / person / time / environment / version / doc_category`
2. 企业知识治理与安全字段
   - `owner_department / doc_type / data_classification / effective_date / authority_level / source_system`
   - `allowed_users / allowed_roles / allowed_departments / project_ids`

这一轮继续补厚了两层：

3. 文档身份与业务语义字段
   - `doc_number / version_status / status / expiry_date`
   - `group_company / subsidiary / plant`
   - `business_domain / process_stage`
   - `equipment_type / equipment_id / system_name / project_name / project_phase`
4. chunk 局部语义字段
   - `section_path / section_level / section_type`
   - `topic_keywords / chunk_summary`
   - `contains_steps / contains_table / contains_contact / contains_version_signal / contains_risk_signal`

当前落地方式：

- `BasicMetadataExtractor`
  - 从标题、文件名、正文中补齐 `doc_number / doc_type / data_classification / effective_date / authority_level`
  - 继续抽取 `version_status / status / business_domain / process_stage / equipment_type / project_name`
  - 对同文档中多个人员角色按优先级选取更核心的 `person`
- `SemanticChunker`
  - 把上述字段稳定写入 `ChunkMetadata.extra`
  - 对 `allowed_departments / allowed_roles / project_ids` 保留列表语义，不再一律转成字符串
  - 同时补 chunk 局部语义，供后续 metadata boost 与更细粒度排序使用
- `metadata_filters`
  - 既支持标量 metadata 匹配，也支持列表型 metadata 匹配
  - 这样 `project_ids / allowed_departments` 才能真正参与 ACL / post-filter
  - Milvus 一级字段也继续扩容，更多企业字段能在服务端先过滤

这层设计的价值是：

1. `structured_filters` 不再只是一组“理想条件”，而是有真实 metadata 可下推
2. ACL fallback 不需要临时拼装字段，可以直接复用统一 metadata
3. `citations` 可以直接展示版本、生效日期、权威级别，而不是生成阶段再二次猜

## Refusal semantics

当前空召回相关的拒答语义已经开始区分两类情况：

1. `no_relevant_chunks`
   说明知识库里没有足够相关的证据
2. `access_denied`
   说明底层其实召回到了候选，但在 ACL / 数据分级过滤后全部不可访问

这两种拒答语义不能混为一谈，因为它们分别对应：

- 数据问题
- 权限问题
