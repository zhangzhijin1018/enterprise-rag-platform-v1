# Xinjiang Energy Knowledge Copilot

当前仓库基于 `enterprise-rag-platform` 演进，目标已收敛为：

> 新疆能源（集团）有限责任公司 企业知识智能副驾

它不再只是通用企业知识库问答 Demo，而是一个面向制度、技术文档、运维 SOP、项目资料、会议纪要等多源知识场景的企业级 RAG 系统，重点强调：

- 可追溯回答
- 权限前置过滤
- 数据分级
- 拒答机制
- 模型路由策略
- 评测闭环与审计语义

面向企业知识库的 **Agentic RAG** 工程化骨架：接入、Redis 热点答案缓存、MySQL FAQ 快速检索、澄清判定、上下文补全、多路查询规划、父子分层切块、混合检索、BGEM3 优先检索编码、Milvus 向量检索、重排、LangGraph 编排、带引用的生成、拒答策略、RAGAS 评测、Prometheus / OpenTelemetry、Docker 与 Kubernetes。

## 要求

- **Python 3.10+**（团队 Conda 示例：`conda activate tmf_project`）
- 可选：`OPENAI_API_KEY`（真实 LLM / RAGAS 评测）；未配置时进入可审计的离线应答模式
- 推荐直接使用 OpenAI-compatible 接口接 `Qwen`：查询理解 / 查询规划默认 `qwen-turbo`，最终回答默认 `qwen-plus`

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
│   ├── retrieval/      # BGEM3 / BM25、dense、hybrid、reranker、Milvus
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
| `POST /chat` | `question`, `conversation_id?`, `history_messages?`, `top_k`, `stream`, 企业用户上下文（可选） |
| `POST /ingest` | 上传 PDF/DOCX/PPTX/HTML/Markdown/TXT/CSV |
| `POST /faq/import` | 导入 FAQ CSV 到 MySQL，并刷新 FAQ 检索索引 |
| `POST /reindex` | 基于 Milvus 中现有 chunk 文本重建稠密向量并 reload 检索器 |
| `POST /eval` | 运行 RAGAS（需密钥） |
| `GET /healthz` | 健康检查 |
| `GET /metrics` | Prometheus |

详见 [docs/api.md](docs/api.md)。

文件类型 parse / chunk 对照说明见 [docs/ingestion_filetype_matrix.md](docs/ingestion_filetype_matrix.md)。
注释覆盖审计见 [docs/comment_coverage_audit.md](docs/comment_coverage_audit.md)。
主链路阅读清单见 [docs/core_reading_paths.md](docs/core_reading_paths.md)。
本地 reranker 微调方案见 [docs/reranker_finetune_guide.md](docs/reranker_finetune_guide.md)。
本地生成模型 QLoRA 微调方案见 [docs/local_llm_finetune_guide.md](docs/local_llm_finetune_guide.md)。

当前入库已覆盖 7 类常见企业文档格式：

- `PDF`：规章、手册、论文、导出报表
- `DOCX`：SOP、制度、方案文档
- `PPTX`：培训课件、汇报材料、架构分享
- `HTML`：知识库网页、帮助中心页面
- `Markdown`：技术文档、FAQ、运行手册
- `TXT`：日志摘录、FAQ 草稿、告警说明
- `CSV`：错误码表、FAQ 导出、配置项清单

当前切块策略已升级为“统一 chunker + 文件类型 profile”：

- `PDF`：优先按页级 section 切分，减少跨页混块
- `DOCX / Markdown / HTML`：继续利用标题层级做结构感知切分
- `PPTX`：使用更紧凑的 slide 级切块参数
- `CSV`：使用更小、更精确的行级 chunk 参数，避免多行表格信息混在一起
- `TXT`：保持通用文本策略，但限制更保守的长度和 overlap

同时 parser 输出也做了增强：

- `DOCX`：保留 heading level、列表项，并补基础表格结构
- `PPTX`：保留 bullet 层级，减少 slide 内短句被拍扁
- `CSV`：每行自动补更强的主键标题，提升行级检索锚点

## 配置

复制 `.env.example` 为 `.env`。当前默认同时启用：

- `Redis`：缓存热点答案与查询改写
- `MySQL`：存储 FAQ 结构化问答
- `Milvus Lite`：执行向量召回

关键变量：

- `REDIS_URL=redis://localhost:6379/0`
- `MYSQL_URL=mysql+pymysql://rag:rag@127.0.0.1:3306/enterprise_rag`
- `VECTOR_BACKEND=milvus`
- `RETRIEVAL_EMBEDDING_BACKEND=bgem3`
- `BGEM3_DEVICE=auto`
- `BGEM3_USE_FP16=true`
- `MILVUS_URI=./data/milvus/enterprise_rag.db`
- `MILVUS_COLLECTION_NAME=rag_chunks`
- `FAQ_BM25_THRESHOLD=0.85`
- `ANSWER_CACHE_TTL_SEC=86400`
- `EMBEDDING_MODEL_NAME=./modes/bge-m3`
- `RERANKER_MODEL_NAME=./modes/bge-reranker-large`
- `FUSION_STRATEGY=rrf`
- `WEIGHTED_FUSION_QUERY_SCENES=policy_lookup,error_code_lookup,structured_fact_lookup`
- `WEIGHTED_FUSION_SCENE_WEIGHTS=policy_lookup:0.65,error_code_lookup:0.75,structured_fact_lookup:0.70`
- `OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
- `QUERY_UNDERSTANDING_VOCAB_PATH=./data/config/query_understanding_vocab.json`
- `LLM_MODEL_NAME=qwen-plus`
- `QUERY_UNDERSTANDING_MODEL_NAME=qwen-turbo`
- `QUERY_PLANNING_MODEL_NAME=qwen-turbo`
- `ANSWER_GENERATION_MODEL_NAME=qwen-plus`
- `QUERY_UNDERSTANDING_CONFIDENCE_THRESHOLD=0.78`
- `QUERY_UNDERSTANDING_FORCE_HYBRID_THRESHOLD=0.58`
- `ENABLE_ACL=true`
- `ENABLE_DATA_CLASSIFICATION=true`
- `ENABLE_MODEL_ROUTING=true`
- `ENABLE_RISK_ENGINE=true`
- `ENABLE_FILE_LOGGING=true`
- `LOG_DIR=./logs`
- `APP_LOG_FILENAME=app.log`
- `AUDIT_LOG_FILENAME=audit.log`
- `RISK_ENGINE_PROVIDER=rule_based`
- `RISK_ENGINE_FAIL_OPEN=true`
- `ENABLE_ML_RISK_HINT=false`
- `ML_RISK_HINT_PROVIDER=disabled`
- `ML_RISK_FAIL_OPEN=true`
- `ML_RISK_REQUEST_STAGE_ENABLED=true`
- `ML_RISK_MODEL_DIR=./modes/ml-risk`
- `ML_RISK_ONNX_PATH=./modes/ml-risk/risk_classifier.onnx`
- `DEFAULT_DATA_CLASSIFICATION=internal`
- `ALLOW_EXTERNAL_LLM_FOR_SENSITIVE=false`
- `LOCAL_ONLY_CLASSIFICATIONS=restricted`
- `ACL_STRICT_MODE=true`
- `AUDIT_LOG_ENABLED=true`

说明：

- `Redis -> MySQL FAQ -> RAG` 是当前问答链路的执行顺序。
- Redis 命中时直接返回热点答案。
- MySQL FAQ 命中且置信度达到阈值时直接返回 FAQ 答案。
- 但**当前版本对带企业安全上下文的请求会绕过 fast path**，原因是 FAQ / cache 层尚未 ACL 化。
- `Milvus` 当前已经成为检索层的唯一权威存储：
  - chunk 原文与 metadata
  - dense 向量
  - sparse 向量
  - 高频 metadata filter 字段的标量倒排索引
  - parent / child 回扩读取
  - `/reindex` 的数据源
- 默认检索编码后端已经切到 `BGEM3` 优先：
  - 入库时会同时生成 `dense + sparse`
  - 两者一起写入 Milvus collection
  - 查询时优先走 Milvus 原生 `hybrid_search`
  - 如果当前环境没有安装 `pymilvus[model]` / `FlagEmbedding`，系统会显式记录降级日志，并临时回退到 `SentenceTransformer + BM25`
- hybrid 检索当前支持“按 `query_scene` 动态选 ranker 策略”：
  - 默认全局策略仍是 `RRF`
  - `policy_lookup / error_code_lookup / structured_fact_lookup` 会自动切到 `WeightedRanker`
  - 并支持按场景指定不同 `sparse_weight`
  - 这样能让制度号、错误码、结构化事实查询更偏向词面命中
- 当前仓库默认直接使用项目内本地模型目录：
  - `./modes/bge-m3`
  - `./modes/bge-reranker-large`
- 这些目录建议只作为**本地模型挂载目录**使用，不要把 `*.bin`、`*.safetensors`、`*.onnx_data`、`data/milvus/*.db` 这类大文件提交到 Git。
- 更稳妥的做法是：代码仓库只保留代码与配置，模型权重通过下载脚本、对象存储或制品仓库单独分发。
- 如果你要切到远端 Milvus Standalone / Cluster，只需要把 `MILVUS_URI` 改成类似 `http://127.0.0.1:19530`。

## 当前企业化安全主链路

当前 `/chat` 已经支持企业用户上下文字段，例如：

- `user_id`
- `username`
- `department`
- `role`
- `project_ids`
- `clearance_level`
- `query_scene`
- `allow_external_llm`

当前企业安全主链路可以概括为：

```text
ChatRequest
-> user_context
-> access_filters
-> request-level risk evaluation
-> LangGraph / retrieval pipeline
-> structured_filters + access_filters
-> sparse / dense retrieval
-> ACL fallback filter(child hits)
-> data_classification
-> model_route
-> retrieval / generation risk evaluation
-> answer / refusal
```

## 当前 Query Understanding 升级

当前 `analyze_query` 已经从“纯规则分流”升级成：

```text
规则信号抽取
-> 规则置信度评估
-> 低置信问题走 LLM 补判
-> 很低置信问题回退 conservative hybrid guardrail
```

当前设计原则：

- 高置信规则：直接走，成本最低、可解释性最好
- 中低置信：交给 `qwen-turbo` 做结构化补判
- 很低置信：强制回到 `hybrid + balanced + no_hyde`，避免把 query 过度路由到某一路

这样做的目标不是替代后续 `clarify / resolve_context / query_plan`，而是让第一层查询理解不再过度死板，同时保持企业系统需要的可控性。

这一轮又把高频业务词从硬编码里抽成了词典层：

- 默认词典文件：[data/config/query_understanding_vocab.json](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/data/config/query_understanding_vocab.json)
- 维护说明：[docs/query_understanding_vocab.md](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/docs/query_understanding_vocab.md)
- 当前运行时不会每次请求重复读取 JSON 文件，而是会把词典预编译成内存索引并缓存复用
- 配置入口：`QUERY_UNDERSTANDING_VOCAB_PATH`

现在后续要补：

- 设备别名
- 部门别名
- 会议纪要关键词
- 项目表达
- 业务域词

优先改词典文件即可，不用先改 `analyze_query.py`。

这一轮又进一步把“新疆能源企业词典”往前做了一层：

- 业务域已扩到 `fuel_management / power_generation / emergency_response / contract_management`
- 部门别名已补 `燃料管理部 / 发电运行部 / 生产技术部 / 招标采购部`
- 场站别名已补 `一号输煤线 / 集控室 / 南露天煤场 / 翻车机卸煤区`
- 系统别名已补 `燃料管控平台 / 生产运营看板 / 设备缺陷管理系统`

- `department_aliases`
- `site_aliases`
- `system_aliases`

也就是说，现在像：

- `安环部`
- `二矿`
- `安生平台`

这类企业内部简称，已经可以直接被归一成：

- `安全环保部`
- `准东二矿`
- `安全生产管理平台`

并进入 `metadata_intent`，后续会继续影响 retrieval filter 和 route 决策。

当前版本已经具备的企业安全语义：

- 检索前 / 检索中 ACL 过滤基础设施
- `access_denied` 与 `no_relevant_chunks` 区分
- 响应返回 `refusal / refusal_reason / answer_mode`
- 响应返回 `data_classification / model_route / audit_id`
- 流式与非流式链路共享同一套企业安全上下文
- request / retrieval / generation 三个执行点共享统一风控接口

## 当前统一风控引擎

这一轮开始，项目把高风险判断从零散规则收敛成了统一风控接口：

- `RiskContext`
- `RiskDecision`
- `RiskEngine`
- `RuleBasedRiskEngine`

当前默认实现仍然是本地规则风控，但主链路已经按“决策点 / 执行点”拆开：

- `/chat`：request-level 风控
- `retrieve_docs_node`：retrieval-level 风控
- `generate_answer_node`：generation-level 风控

当前又新增了一层“ML 风控 hint”接入位，但仍然坚持：

- ML 只输出 `risk_level_hint`
- `RuleBasedRiskEngine` 仍负责最终 `allow / deny / local_only / minimize`
- 冲突时默认取更高风险
- 第一轮只在 request-level 接入，默认关闭，开启失败时回退到纯规则模式

当前默认策略包括：

- 明显的批量敏感导出请求：入口直接拒答
- `restricted / local_only`：强制本地受限模式
- `sensitive`：最小必要上下文 + 脱敏
- `internal`：默认脱敏后外发
- `access_denied`：明确拒答，而不是继续生成

这样后续如果要接外部 PDP，例如企业风控中心、OPA、策略网关，不需要重写主链路，只需要替换 `RiskEngine` 的实现。

## 当前企业 metadata 与引用语义

这一轮开始，入库链路不再只补通用检索字段，而是开始统一企业知识文档的 metadata 语义。当前 metadata extractor 和 chunker 会优先补齐并透传这些字段：

- 文档身份与治理字段：`doc_number / version / version_status / status / effective_date / expiry_date / doc_category / doc_type / authority_level / data_classification / source_system`
- 组织与责任字段：`group_company / subsidiary / plant / department / owner_department / issued_by / approved_by / owner_role`
- 业务语义字段：`business_domain / process_stage / applicable_region / applicable_site / equipment_type / equipment_id / system_name / project_name / project_phase`
- 原有检索字段：`shift / line / person / time / environment`
- 列表字段：`allowed_users / allowed_roles / allowed_departments / project_ids`

当前策略说明：

- `BasicMetadataExtractor` 现在会从标题、文件名、正文里尽量补 `doc_number / doc_type / data_classification / effective_date / authority_level / version_status / business_domain / process_stage / equipment_type / project_name`。
- `SemanticChunker` 现在除了透传文档级字段，还会补 chunk 局部语义：
  - `section_path / section_level / section_type`
  - `topic_keywords / chunk_summary`
  - `contains_steps / contains_table / contains_contact / contains_version_signal / contains_risk_signal`
- `metadata_filters` 已支持列表型 metadata 匹配，因此 `project_ids / allowed_departments / allowed_roles` 这类 ACL 字段现在可以真正参与 post-filter。
- Milvus collection 这一轮也继续扩了一级字段，`doc_number / owner_department / doc_type / data_classification / effective_date / authority_level / business_domain / process_stage / equipment_type / project_name / section_type` 等字段已经能优先服务端过滤。

对应地，当前 `/chat` 返回的 `citations` 也不再只有 `doc_id / chunk_id / title / source / page / section`，还会尽量补充：

- `doc_type`
- `owner_department`
- `data_classification`
- `version`
- `effective_date`
- `authority_level`
- `source_system`

这意味着当前回答里的引用已经开始具备“文档类型、归属部门、版本、生效日期、权威级别”的企业知识治理语义，而不只是一个可跳转 chunk。

## 当前 retrieval 质量与性能优化

这一轮开始，检索主链路不再只是“多路 query + hybrid retrieval”，而是开始让 query intent 和企业 metadata 真正参与性能优化。

当前新增了 4 个关键点：

- `analyze_query_node` 会额外输出 `query_scene / preferred_retriever / top_k_profile / metadata_intent`
- `retrieve_docs_node` 会把 `metadata_intent + structured_filters + access_filters` 合并成最终 retrieval filters
- sparse / dense 的 `top_k` 会按 `top_k_profile` 和 `preferred_retriever` 动态调整，而不是每一路都跑满默认值
- fusion 之后会按 metadata 命中情况做轻量 `metadata boost`，再进入 rerank
- 对 `department / plant / system_name / business_domain` 这类企业实体归一结果，当前会额外叠加 `enterprise entity boost`
- 对 `precise / sparse` 问题会主动裁掉低价值 route，例如多余 `sub_query / hyde`

当前这套优化的目标不是“把规则写死”，而是：

- 精确问题少召回一些无关候选
- 复杂问题保留更宽召回面
- 让 Milvus 和本地 filter 更早吃到企业 metadata
- 减少无效 rerank 候选和无效上下文 token

与此同时，`rerank_docs_node` 现在会先按 `rerank_candidate_multiplier / rerank_candidate_max` 裁剪候选，再交给 cross-encoder。

生成侧这一轮也补了一层 token 优化：

- 生成前会做 `context packing`
- 限制参与 prompt 的文档数、单文档 chunk 数和总字符数
- 同文档重复 section 不会重复喂给模型

同时，citation 现在开始对外暴露更强的解释性字段：

- `business_domain / process_stage / section_path`
- `matched_routes`
- `retrieval_score / semantic_score / governance_rank_score`
- `selection_reason`

这样前端和面试演示时，不只知道“引用了什么”，也能知道“为什么系统优先选了它”。

这一轮又把这些解释性信号接进了评测摘要：

- `matched_route:*`
- `entity_match:*`
- `metadata_boost_hit_rate`
- `enterprise_entity_boost_hit_rate`
- `governance_boost_hit_rate`
- `avg_explainable_citations`

这样 `/eval` 不只看答案分数，也能看 retrieval 优化和证据解释到底有没有真实生效。

这一轮继续把企业实体归一真正接进 retrieval 排序：

- `department / owner_department` 会合并成统一的 `department` 实体组
- `plant / applicable_site` 会合并成统一的 `site` 实体组
- `system_name / business_domain / process_stage / equipment_* / project_name` 会作为企业实体组单独提权
- 这些实体组 boost 会写进 trace，后续 citation explainability 和 `/eval` 摘要都能直接看到
- 同时 retrieval filter 现在对 `department / owner_department`、`plant / applicable_site` 采用“组内 OR、组间 AND”的语义，避免企业别名归一后把过滤条件收得过死

这一轮又把 query understanding 信号接进了评测闭环：

- `analysis_confidence`
- `analysis_source`
- `analysis_reason`
- `avg_analysis_confidence`
- `analysis_source:*`

这样 badcase 分析时可以直接看出：

- 哪些题是高置信规则直走
- 哪些题依赖 `qwen-turbo` 做结构化补判
- 哪些题最终被 guardrail 强制回退到保守 `hybrid`

同时评测 JSON 里还新增了 `query_understanding_report`，Markdown explainability report 也会多一节 `Query Understanding Tuning`，直接汇总：

- 高频 `query_scene`
- 高频 `guardrail` 场景
- 高频 `llm_enhanced` 场景
- 面向下一轮调参的建议

同时 `/eval` 现在会额外导出一份 Markdown explainability report，专门用于：

- badcase 回放
- route / boost / governance 生效分析
- 面试演示时快速挑出代表性问题

这样当前主链路已经具备两层明显的性能收益：

这一轮又补了一层“源码可读性”增强：

- `core/models/document.py` 的核心字段已补中文说明
- `apps/api/schemas/chat.py` 的请求/响应字段已补中文 description
- `core/orchestration/state.py` 的状态字段已补中文行内注释
- `core/retrieval/milvus_retriever.py` 已补 Milvus schema 字段中文说明
- `core/config/settings.py`、`core/generation/citation_formatter.py`、`core/retrieval/metadata_filters.py` 也已补中文字段说明与注释
- `core/retrieval/schemas.py`、`core/retrieval/governance.py`、`core/orchestration/nodes/retrieve_docs.py` 这一条“检索 -> 融合 -> 治理排序”主链路也已补中文说明
- `core/orchestration/query_understanding_vocab.py`、`core/orchestration/nodes/analyze_query.py`、`core/orchestration/query_expansion.py` 这一条“query understanding -> query planning”主链路也已补中文说明
- `core/generation/context_format.py`、`core/orchestration/nodes/generate_answer.py`、`core/generation/answer_builder.py` 这一条“上下文拼接 -> 生成 -> 引用解析”主链路也已补中文说明
- `core/security/risk_engine.py`、`core/observability/audit.py`、`core/generation/egress_policy.py` 这一条“风控 -> 审计 -> 出域控制”主链路也已补中文说明
- `core/ingestion/metadata_extractors/basic.py`、`core/ingestion/chunkers/semantic_chunker.py`、`core/ingestion/pipeline.py` 这一条“metadata 抽取 -> 分层切块 -> 入库索引”主链路也已补中文说明
- `core/orchestration/graph.py`、`core/orchestration/fast_path.py`、`core/generation/llm_client.py`、`core/retrieval/dense_retriever.py`、`core/retrieval/sparse_retriever.py`、`core/retrieval/hybrid_fusion.py`、`core/retrieval/milvus_retriever.py`、`core/retrieval/faq_store.py`、`core/retrieval/faq_retriever.py`、`core/retrieval/cache.py`、`apps/api/dependencies/common.py`、`apps/api/routes/ingest.py`、`apps/api/routes/eval.py`、`core/services/runtime.py` 这些运行时与支撑模块也已补中文说明
- `core/observability/metrics.py`、`core/observability/tracing.py`、`apps/api/routes/health.py`、`apps/api/routes/faq.py`、`apps/worker/jobs/ingest_job.py`、`apps/api/schemas/common.py`、`apps/api/schemas/faq.py`、`apps/api/schemas/eval_schema.py` 这些边缘但常用的支撑模块也已补中文说明

- 检索层：更少无效召回
- 重排层：更少无效精排

## 当前企业治理排序与冲突提示

这一轮开始，系统会在 cross-encoder 语义重排之后，再叠一层轻量的企业治理优先级：

- `authority_level` 更高的证据优先
- `effective_date` 更新的证据优先
- `version` 更新的证据优先

当前策略不是替代语义相关性，而是在“语义已经足够相关”的候选里做更企业化的顺序调整。系统会把这部分结果写入 trace，例如：

- `semantic_score`
- `governance_bonus`
- `governance_rank_score`

同时，当前系统已经支持对最终上下文做保守型冲突提示：

- 如果同主题证据之间出现版本不一致
- 或生效日期不一致
- 或权威级别不同

则会返回：

- `conflict_detected`
- `conflict_summary`

并在生成阶段把这条治理提示一并喂给模型，避免模型把“多版本、多权威来源”偷偷融合成一个看似流畅但不可追溯的答案。

当前前端控制台也已经把这些治理信号展示出来：

- 回答区会显示 `answer_mode / data_classification / model_route`
- 如果触发冲突，会单独展示 `conflict_summary`
- 如果触发拒答，会展示 `refusal_reason`
- 页面保留 `audit_id`，便于和日志、评测报告对照

## 当前数据分级执行层

这一轮之后，`data_classification` 不再只是标签，已经开始真实影响生成链路：

- `public`
  允许完整上下文进入外部模型
- `internal`
  默认对常见敏感模式做脱敏后再进入外部模型
- `sensitive`
  只允许“脱敏 + 最小必要上下文”出域，当前默认限制为少量片段和较短上下文
- `restricted`
  禁止把原始上下文发给外部模型；当前如果没有本地模型执行器，会直接拒答

当前实现位置：

- [core/generation/egress_policy.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/generation/egress_policy.py)
- [core/orchestration/nodes/generate_answer.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/nodes/generate_answer.py)
- [apps/api/routes/chat.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/apps/api/routes/chat.py)

当前还没做的是：

- 真正的本地模型执行层
- 更精细的字段级脱敏策略
- L2/L3 的业务自定义出域白名单

不过这一轮已经补了一个很重要的过渡层：

- `restricted / local_only`
  当前不再一律拒答
- 如果开启 `ENABLE_LOCAL_FALLBACK_GENERATION`
  系统会走“本地受限模式占位执行”
- 它不会调用外部模型，而是基于已授权上下文生成一个可引用、可审计的本地占位答案

## 当前审计与日志治理

这一轮开始，系统已经不只返回 `audit_id`，而是有了统一的审计与链路日志结构：

- 本地业务链路日志：`logs/app.log`
- 本地审计/安全事件日志：`logs/audit.log`
- 每个请求统一生成 `trace_id`
- `/chat` 响应头会回传 `X-Trace-ID`

当前关键步骤日志默认以 `INFO` 级别写入 `.log` 文件，便于按一次请求回放：

- `request_received`
- `request_denied`
- `graph_started / graph_completed`
- `retrieval_started / retrieval_completed`
- `generation_started / generation_completed`
- `response_sent`

当前前端问答结果区也会直接展示：

- `trace_id`
- `audit_id`

这样你拿到一条回答后，可以马上按 `trace_id` 去 grep `logs/app.log` 或 `logs/audit.log`。

- 请求审计：`request_received`
- prompt 审计：`prompt_audited`
- 输出审计：`output_audited`
- 响应审计：`response_sent`

当前审计事件会记录：

- `trace_id`
- `audit_id`
- `risk_level`
- `risk_action / risk_reason`
- `question_hash / question_preview`
- `user_id / department / role / project_ids`
- `data_classification / model_route`
- `refusal / refusal_reason`
- `conflict_detected / conflict_summary`
- `retrieved_chunk_ids / retrieved_doc_ids`
- `prompt_hash / prompt_preview`
- `output_hash / output_preview`

其中：

- `app.log` 更偏排障，适合按 `trace_id` 看用户这次请求走到了哪些关键步骤
- `audit.log` 更偏审计与安全留痕，保留结构化审计事件
- `preview` 默认会经过脱敏，避免把原始敏感内容直接落日志

当前还新增了 `security_alert` 分流事件，典型触发条件包括：

- 高风险问题
- `restricted` 数据
- 权限拒答
- 冲突检测命中
- 风控引擎显式要求告警

这样当前链路已经从“只有审计记录”进一步升级成“有可分流的安全告警信号”。

## 当前评测报告新增的治理信号

当前 `POST /eval` 生成的报告，除了 RAGAS 指标外，也会把本轮问答的治理信号写进报告：

- `refusal / refusal_reason`
- `answer_mode`
- `data_classification`
- `model_route`
- `conflict_detected / conflict_summary`
- `audit_id`

同时 `summary` 里会额外统计：

- `sample_count`
- `refusal_rate`
- `conflict_detected_rate`
- `classification:*`
- `model_route:*`

这样当前评测就不再只是“答案像不像对”，而是开始能观察企业知识副驾的安全与治理行为。

当前默认评测集已经切到：

- `core/evaluation/datasets/enterprise_eval.jsonl`

它比最小样例更贴近面试和企业场景，已经开始覆盖：

- 制度问答
- SOP 对比
- 多版本冲突
- 权限拒答
- 无依据拒答
- 多轮承接问题

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

## 当前查询编排

当前主链路已经从“固定问题分类驱动”升级为“策略信号驱动”：

```text
question
-> analyze_signals
-> clarify_gate
   -> need_clarify ? ask_clarify : continue
-> resolve_context
-> build_query_plan
-> retrieve
-> rerank
-> generate
-> validate
```

设计重点：

- 不再依赖 `error_code / procedure / general` 这类 demo 风格三分类
- `clarify` 是检索前闸门，围绕缺槽位判断是否先追问
- `resolve_context` 专门处理多轮对话里的代词、省略和承接问题
- `/chat` 已支持可选 `history_messages`，前端传入最近几轮对话时，多轮补全效果更稳定
- `build_query_plan` 基于 `strategy_signals` 决定是否启用 `resolved / rewrite / keyword / sub_query / hyde`
- `structured_filters` 会随检索请求下传，便于后续接 Milvus metadata filter / SQL / 排班系统
- Milvus collection 已预留 `department / shift / line / person / time / environment / version / doc_category` 一级字段，相关过滤条件会优先服务端下推；旧 collection 在有本地快照时可自动重建升级

当前重点支持的企业问答场景：

- 制度规范、SOP、操作手册
- 设备故障、IT 系统问题、原因与处理类问题
- 时间 / 部门 / 班次 / 人员类事实查询
- 多轮上下文追问与复杂问题多路召回

当前仍在持续补齐的企业化能力：

- ingestion 端更完整的企业 metadata schema 与数据迁移
- FAQ / cache 层 ACL 化
- 更强的文档版本、权威级别和冲突检测
- `model_route` 对应的真实多模型执行层
