"""全局配置模块。

本项目通过 `pydantic-settings` 统一读取 `.env` 与环境变量，
这样 API、Worker、检索器、评测器可以共享同一套配置来源。
"""

from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """项目运行时配置。

    设计思路：
    1. 用类型约束保证配置在启动阶段就暴露问题。
    2. 用 alias 对接环境变量，避免代码层到处硬编码字符串。
    3. 所有子模块都只依赖这个对象，不直接读 os.environ。
    """

    # `model_config` 告诉 pydantic-settings 去哪里找 `.env`，以及遇到额外字段时如何处理。
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 运行环境与 API 服务基础配置。
    app_env: Literal["development", "staging", "production"] = Field(
        default="development", alias="APP_ENV", description="运行环境"
    )
    project_name: str = Field(
        default="Xinjiang Energy Knowledge Copilot",
        alias="PROJECT_NAME",
        description="项目名称",
    )
    api_host: str = Field(default="0.0.0.0", alias="API_HOST", description="API 监听地址")
    api_port: int = Field(default=8000, alias="API_PORT", description="API 监听端口")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL", description="日志级别")
    enable_file_logging: bool = Field(default=True, alias="ENABLE_FILE_LOGGING", description="是否启用本地日志落盘")
    log_dir: str = Field(default="./logs", alias="LOG_DIR", description="日志目录")
    app_log_filename: str = Field(default="app.log", alias="APP_LOG_FILENAME", description="应用主日志文件名")
    audit_log_filename: str = Field(default="audit.log", alias="AUDIT_LOG_FILENAME", description="审计日志文件名")
    log_max_bytes: int = Field(default=10 * 1024 * 1024, alias="LOG_MAX_BYTES", description="单个日志文件最大字节数")
    log_backup_count: int = Field(default=5, alias="LOG_BACKUP_COUNT", description="日志轮转保留份数")

    # 基础设施依赖。
    redis_url: str | None = Field(default="redis://localhost:6379/0", alias="REDIS_URL", description="Redis 连接串")
    query_understanding_vocab_path: str = Field(
        default="./data/config/query_understanding_vocab.json",
        alias="QUERY_UNDERSTANDING_VOCAB_PATH",
        description="查询理解词典文件路径",
    )
    mysql_url: str | None = Field(
        default="mysql+pymysql://rag:rag@127.0.0.1:3306/enterprise_rag",
        alias="MYSQL_URL",
        description="MySQL 连接串",
    )

    # 向量后端配置。
    #
    # 第五轮增强开始，项目支持两种 dense backend：
    #
    # 当前主线统一收敛到 `milvus`：
    #
    # - Milvus / Milvus Lite 作为正式向量检索后端
    # - Milvus 同时承担 chunk 原文、metadata 和 dense 向量的权威存储
    # - 上层仍然保留统一检索接口，便于后续继续演进
    #
    # 这里默认切到 `milvus`，并优先使用 Milvus Lite 的本地文件 URI，
    # 这样在不额外起服务的情况下，也能直接体验 Milvus 的集合、schema、
    # 向量索引和检索 API；后续如果切到远端 Standalone / Distributed，
    # 只需要把 URI 改成 `http://host:19530` 这类地址即可。
    vector_backend: Literal["milvus"] = Field(
        default="milvus",
        alias="VECTOR_BACKEND",
        description="向量检索后端类型",
    )
    retrieval_embedding_backend: Literal["classic", "bgem3"] = Field(
        default="bgem3",
        alias="RETRIEVAL_EMBEDDING_BACKEND",
        description="检索编码后端：classic 表示 SentenceTransformer+BM25，bgem3 表示 BGEM3 dense+sparse",
    )
    bgem3_device: str = Field(
        default="auto",
        alias="BGEM3_DEVICE",
        description="BGEM3 运行设备，支持 auto/cpu/cuda/mps",
    )
    bgem3_use_fp16: bool = Field(
        default=True,
        alias="BGEM3_USE_FP16",
        description="BGEM3 是否优先使用 FP16",
    )
    milvus_uri: str = Field(
        default="./data/milvus/enterprise_rag.db",
        alias="MILVUS_URI",
        description="Milvus / Milvus Lite 连接地址",
    )
    milvus_token: str | None = Field(default=None, alias="MILVUS_TOKEN", description="Milvus token")
    milvus_db_name: str | None = Field(default=None, alias="MILVUS_DB_NAME", description="Milvus 数据库名")
    milvus_collection_name: str = Field(
        default="rag_chunks",
        alias="MILVUS_COLLECTION_NAME",
        description="Milvus collection 名称",
    )
    milvus_index_type: str = Field(default="AUTOINDEX", alias="MILVUS_INDEX_TYPE", description="Milvus 索引类型")
    milvus_metric_type: Literal["COSINE", "IP", "L2"] = Field(
        default="COSINE",
        alias="MILVUS_METRIC_TYPE",
        description="Milvus 向量距离类型",
    )

    # LLM 服务配置。
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY", description="OpenAI-compatible API Key")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL", description="OpenAI-compatible Base URL"
    )

    # 模型选择：
    # - `llm_model_name` 用于回答生成 / 查询改写
    # - `embedding_model_name` 用于把文本映射成向量
    # - `reranker_model_name` 用于二次精排
    #
    # 这里优先给中文知识库一个更稳妥的默认值：
    # 1. 本项目的示例文档、注释、接口说明都以中文为主；
    # 2. 企业 FAQ / SOP / 工单 / 制度类问答通常也是中文或中英混合；
    # 3. 如果继续使用英文通用 MiniLM 作为默认 embedding / reranker，
    #    在中文场景里经常会出现“能跑通，但召回不够准、重排不够稳”的问题。
    #
    # 因此这里把默认模型切到中文场景里更常见的 BGE 路线。
    # 如果你的知识库主要是英文，或者你已经有团队统一模型，
    # 仍然可以通过环境变量覆盖，不会影响接口和主链路。
    llm_model_name: str = Field(default="qwen-plus", alias="LLM_MODEL_NAME", description="通用 LLM 默认模型名")
    query_understanding_model_name: str = Field(
        default="qwen-turbo",
        alias="QUERY_UNDERSTANDING_MODEL_NAME",
        description="查询理解模型名",
    )
    query_planning_model_name: str = Field(
        default="qwen-turbo",
        alias="QUERY_PLANNING_MODEL_NAME",
        description="查询规划模型名",
    )
    answer_generation_model_name: str = Field(
        default="qwen-plus",
        alias="ANSWER_GENERATION_MODEL_NAME",
        description="最终回答生成模型名",
    )
    embedding_model_name: str = Field(
        # `BAAI/bge-m3` 支持中文、多语种和较长文本检索，
        # 对 FAQ、SOP、产品文档、工单经验库这类典型企业知识库更友好。
        #
        # 当前仓库默认优先使用项目内本地模型目录 `./modes/bge-m3`，
        # 这样在离线或内网环境下也能直接跑起 BGEM3 检索。
        # 如果后续你要切回 HuggingFace 名称或别的本地路径，仍然可以通过环境变量覆盖。
        default="./modes/bge-m3",
        alias="EMBEDDING_MODEL_NAME",
        description="embedding 模型名",
    )
    reranker_model_name: str = Field(
        # reranker 负责在召回候选上做最后一轮精排，
        # 它对“最终给 LLM 喂哪些 chunk”影响非常大。
        # 这里同样优先采用中文检索生态里更常见的 BGE reranker 默认值。
        #
        # 当前仓库默认优先使用项目内本地模型目录 `./modes/bge-reranker-large`，
        # 这样交叉编码器精排也能直接走本地模型，不依赖外网下载。
        default="./modes/bge-reranker-large",
        alias="RERANKER_MODEL_NAME",
        description="reranker 模型名",
    )

    # 检索链路的召回 / 融合 / 重排参数。
    bm25_top_k: int = Field(default=20, alias="BM25_TOP_K", description="BM25 默认 top_k")
    dense_top_k: int = Field(default=20, alias="DENSE_TOP_K", description="dense 检索默认 top_k")
    hybrid_top_k: int = Field(default=30, alias="HYBRID_TOP_K", description="hybrid 融合候选上限")
    rerank_top_n: int = Field(default=8, alias="RERANK_TOP_N", description="重排后保留数量")
    rerank_candidate_multiplier: int = Field(
        default=4,
        alias="RERANK_CANDIDATE_MULTIPLIER",
        description="rerank 候选扩张倍数",
    )
    rerank_candidate_max: int = Field(default=24, alias="RERANK_CANDIDATE_MAX", description="rerank 候选最大数量")
    broad_sub_query_limit: int = Field(default=3, alias="BROAD_SUB_QUERY_LIMIT", description="broad 场景子查询上限")
    balanced_sub_query_limit: int = Field(default=2, alias="BALANCED_SUB_QUERY_LIMIT", description="balanced 场景子查询上限")
    keyword_route_limit: int = Field(default=2, alias="KEYWORD_ROUTE_LIMIT", description="关键词 route 上限")
    query_understanding_confidence_threshold: float = Field(
        default=0.78,
        alias="QUERY_UNDERSTANDING_CONFIDENCE_THRESHOLD",
        description="query understanding 触发 LLM 增强阈值",
    )
    query_understanding_force_hybrid_threshold: float = Field(
        default=0.58,
        alias="QUERY_UNDERSTANDING_FORCE_HYBRID_THRESHOLD",
        description="query understanding 强制回退 hybrid 阈值",
    )
    query_understanding_max_tokens: int = Field(
        default=512,
        alias="QUERY_UNDERSTANDING_MAX_TOKENS",
        description="query understanding LLM 最大输出 token",
    )
    metadata_match_boost: float = Field(default=0.08, alias="METADATA_MATCH_BOOST", description="通用 metadata 提权分数")
    enterprise_entity_match_boost: float = Field(
        default=0.12,
        alias="ENTERPRISE_ENTITY_MATCH_BOOST",
        description="企业实体命中额外提权分数",
    )
    structured_filter_boost: float = Field(default=0.05, alias="STRUCTURED_FILTER_BOOST", description="显式结构化过滤命中提权分数")
    fusion_strategy: Literal["rrf", "weighted"] = Field(
        default="rrf", alias="FUSION_STRATEGY", description="融合策略"
    )
    fusion_sparse_weight: float = Field(default=0.5, alias="FUSION_SPARSE_WEIGHT", description="weighted fusion 下 sparse 权重")
    weighted_fusion_query_scenes: str = Field(
        default="policy_lookup,error_code_lookup,structured_fact_lookup",
        alias="WEIGHTED_FUSION_QUERY_SCENES",
        description="命中这些 query_scene 时优先切到 weighted fusion，多个值用逗号分隔",
    )
    weighted_fusion_scene_weights: str = Field(
        default="policy_lookup:0.65,error_code_lookup:0.75,structured_fact_lookup:0.70",
        alias="WEIGHTED_FUSION_SCENE_WEIGHTS",
        description="按 query_scene 指定 weighted fusion 的 sparse 权重，格式 scene:weight，多组用逗号分隔",
    )
    faq_bm25_threshold: float = Field(default=0.85, alias="FAQ_BM25_THRESHOLD", description="FAQ 命中阈值")
    faq_top_k: int = Field(default=3, alias="FAQ_TOP_K", description="FAQ 检索 top_k")
    answer_cache_ttl_sec: int = Field(default=86400, alias="ANSWER_CACHE_TTL_SEC", description="答案缓存 TTL（秒）")
    faq_seed_path: str = Field(default="./data/mock_corpus/faq_seed.csv", alias="FAQ_SEED_PATH", description="FAQ 种子数据路径")

    # 拒答门槛：
    # - `min_retrieval_score` 用在召回层的最低可用阈值
    # - `min_rerank_score` 用在生成前的最终相关性阈值
    # - `refusal_confidence_threshold` 用在后处理阶段判断是否该拒答
    min_retrieval_score: float = Field(default=0.15, alias="MIN_RETRIEVAL_SCORE", description="召回最低可用阈值")
    min_rerank_score: float = Field(default=-5.0, alias="MIN_RERANK_SCORE", description="重排最低可用阈值")
    refusal_confidence_threshold: float = Field(
        default=0.35, alias="REFUSAL_CONFIDENCE_THRESHOLD", description="拒答置信度阈值"
    )

    # 可观测性配置：如果提供 OTLP 端点，就会把 trace 导出到收集器。
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT", description="OTLP 导出地址"
    )
    otel_service_name: str = Field(
        default="enterprise-rag-api", alias="OTEL_SERVICE_NAME", description="OTel service name"
    )

    # 评测配置：样例数据集和报告输出目录。
    eval_output_dir: str = Field(default="./data/eval_reports", alias="EVAL_OUTPUT_DIR", description="评测报告输出目录")
    eval_dataset_path: str = Field(
        default="./core/evaluation/datasets/enterprise_eval.jsonl", alias="EVAL_DATASET_PATH", description="默认评测集路径"
    )

    # 前端开发模式与跨域部署都依赖这组来源配置。
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ORIGINS",
        description="允许跨域来源",
    )

    # 企业化安全与路由配置。
    enable_acl: bool = Field(default=True, alias="ENABLE_ACL", description="是否启用 ACL")
    enable_data_classification: bool = Field(
        default=True,
        alias="ENABLE_DATA_CLASSIFICATION",
        description="是否启用数据分级",
    )
    enable_model_routing: bool = Field(default=True, alias="ENABLE_MODEL_ROUTING", description="是否启用模型路由")
    enable_risk_engine: bool = Field(default=True, alias="ENABLE_RISK_ENGINE", description="是否启用风控引擎")
    risk_engine_provider: Literal["rule_based"] = Field(
        default="rule_based",
        alias="RISK_ENGINE_PROVIDER",
        description="风控引擎提供方",
    )
    risk_engine_fail_open: bool = Field(default=True, alias="RISK_ENGINE_FAIL_OPEN", description="风控失败时是否放行")
    enable_ml_risk_hint: bool = Field(
        default=False,
        alias="ENABLE_ML_RISK_HINT",
        description="是否启用 ML 风控 hint",
    )
    ml_risk_hint_provider: Literal["disabled", "mock", "onnx"] = Field(
        default="disabled",
        alias="ML_RISK_HINT_PROVIDER",
        description="ML 风控 hint 提供方",
    )
    ml_risk_fail_open: bool = Field(
        default=True,
        alias="ML_RISK_FAIL_OPEN",
        description="ML 风控 hint 失败时是否回退到纯规则模式",
    )
    ml_risk_request_stage_enabled: bool = Field(
        default=True,
        alias="ML_RISK_REQUEST_STAGE_ENABLED",
        description="是否在请求阶段启用 ML 风控 hint",
    )
    ml_risk_model_dir: str = Field(
        default="./modes/ml-risk",
        alias="ML_RISK_MODEL_DIR",
        description="ML 风控模型目录",
    )
    ml_risk_onnx_path: str = Field(
        default="./modes/ml-risk/risk_classifier.onnx",
        alias="ML_RISK_ONNX_PATH",
        description="ML 风控 ONNX 模型路径",
    )
    enable_governance_ranking: bool = Field(default=True, alias="ENABLE_GOVERNANCE_RANKING", description="是否启用治理排序")
    enable_conflict_detection: bool = Field(default=True, alias="ENABLE_CONFLICT_DETECTION", description="是否启用冲突检测")
    default_data_classification: Literal["public", "internal", "sensitive", "restricted"] = (
        Field(default="internal", alias="DEFAULT_DATA_CLASSIFICATION", description="默认数据分级")
    )
    internal_redact_for_external: bool = Field(
        default=True,
        alias="INTERNAL_REDACT_FOR_EXTERNAL",
        description="internal 数据出域前是否脱敏",
    )
    generation_context_max_docs: int = Field(
        default=4,
        alias="GENERATION_CONTEXT_MAX_DOCS",
        description="生成前最多保留的文档数",
    )
    generation_context_max_chunks_per_doc: int = Field(
        default=2,
        alias="GENERATION_CONTEXT_MAX_CHUNKS_PER_DOC",
        description="单文档最多保留的 chunk 数",
    )
    generation_context_max_chars: int = Field(
        default=3200,
        alias="GENERATION_CONTEXT_MAX_CHARS",
        description="生成前上下文最大字符数",
    )
    enable_local_fallback_generation: bool = Field(
        default=True,
        alias="ENABLE_LOCAL_FALLBACK_GENERATION",
        description="是否启用本地占位生成",
    )
    sensitive_context_max_chunks: int = Field(
        default=1,
        alias="SENSITIVE_CONTEXT_MAX_CHUNKS",
        description="sensitive 数据最多出域 chunk 数",
    )
    sensitive_context_max_chars: int = Field(
        default=600,
        alias="SENSITIVE_CONTEXT_MAX_CHARS",
        description="sensitive 数据最多出域字符数",
    )
    authority_priority_boost: float = Field(default=0.08, alias="AUTHORITY_PRIORITY_BOOST", description="权威级别排序加权")
    freshness_priority_boost: float = Field(default=0.06, alias="FRESHNESS_PRIORITY_BOOST", description="新鲜度排序加权")
    version_priority_boost: float = Field(default=0.04, alias="VERSION_PRIORITY_BOOST", description="版本排序加权")
    conflict_detection_top_k: int = Field(default=5, alias="CONFLICT_DETECTION_TOP_K", description="冲突检测参与 top_k")
    allow_external_llm_for_sensitive: bool = Field(
        default=False,
        alias="ALLOW_EXTERNAL_LLM_FOR_SENSITIVE",
        description="sensitive 数据是否允许外部模型",
    )
    local_only_classifications: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["restricted"],
        alias="LOCAL_ONLY_CLASSIFICATIONS",
        description="只允许本地处理的数据分级",
    )
    acl_strict_mode: bool = Field(default=True, alias="ACL_STRICT_MODE", description="ACL 是否严格模式")
    audit_log_enabled: bool = Field(default=True, alias="AUDIT_LOG_ENABLED", description="是否启用审计日志")
    audit_log_redact_content: bool = Field(default=True, alias="AUDIT_LOG_REDACT_CONTENT", description="审计日志是否脱敏")
    audit_log_preview_chars: int = Field(default=240, alias="AUDIT_LOG_PREVIEW_CHARS", description="审计预览截断长度")
    alert_on_high_risk_queries: bool = Field(default=True, alias="ALERT_ON_HIGH_RISK_QUERIES", description="是否告警高风险问题")
    alert_on_restricted_access: bool = Field(default=True, alias="ALERT_ON_RESTRICTED_ACCESS", description="是否告警 restricted 访问")
    alert_on_conflict_detected: bool = Field(default=True, alias="ALERT_ON_CONFLICT_DETECTED", description="是否告警冲突检测")
    supported_departments: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        alias="SUPPORTED_DEPARTMENTS",
        description="允许的部门白名单",
    )
    supported_roles: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        alias="SUPPORTED_ROLES",
        description="允许的角色白名单",
    )

    @field_validator(
        "local_only_classifications",
        "supported_departments",
        "supported_roles",
        mode="before",
    )
    @classmethod
    def _split_csv_values(cls, value: Any) -> Any:
        """兼容通过环境变量传逗号分隔的列表配置。"""

        if value is None:
            return value
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """返回进程级单例配置。

    `lru_cache` 的作用：
    - 避免每次调用都重复解析环境变量。
    - 保证全进程拿到的是同一份配置对象，便于调试和测试。
    """

    return Settings()
