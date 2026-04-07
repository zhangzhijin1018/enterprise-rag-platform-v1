"""全局配置模块。

本项目通过 `pydantic-settings` 统一读取 `.env` 与环境变量，
这样 API、Worker、检索器、评测器可以共享同一套配置来源。
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        default="development", alias="APP_ENV"
    )
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # 基础设施依赖。
    redis_url: str | None = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    vector_store_path: str = Field(default="./data/vector_store", alias="VECTOR_STORE_PATH")
    mysql_url: str | None = Field(
        default="mysql+pymysql://rag:rag@127.0.0.1:3306/enterprise_rag",
        alias="MYSQL_URL",
    )

    # 向量后端配置。
    #
    # 第五轮增强开始，项目支持两种 dense backend：
    #
    # 1. `file`
    #    - 继续沿用本地 `embeddings.npy + NumPy` 检索
    #    - 适合教学、最小依赖、本地快速联调
    #
    # 2. `milvus`
    #    - 使用 Milvus / Milvus Lite 作为正式向量检索后端
    #    - 更贴近生产环境里的向量数据库形态
    #
    # 这里默认切到 `milvus`，并优先使用 Milvus Lite 的本地文件 URI，
    # 这样在不额外起服务的情况下，也能直接体验 Milvus 的集合、schema、
    # 向量索引和检索 API；后续如果切到远端 Standalone / Distributed，
    # 只需要把 URI 改成 `http://host:19530` 这类地址即可。
    vector_backend: Literal["file", "milvus"] = Field(
        default="milvus",
        alias="VECTOR_BACKEND",
    )
    milvus_uri: str = Field(
        default="./data/milvus/enterprise_rag.db",
        alias="MILVUS_URI",
    )
    milvus_token: str | None = Field(default=None, alias="MILVUS_TOKEN")
    milvus_db_name: str | None = Field(default=None, alias="MILVUS_DB_NAME")
    milvus_collection_name: str = Field(
        default="rag_chunks",
        alias="MILVUS_COLLECTION_NAME",
    )
    milvus_index_type: str = Field(default="AUTOINDEX", alias="MILVUS_INDEX_TYPE")
    milvus_metric_type: Literal["COSINE", "IP", "L2"] = Field(
        default="COSINE",
        alias="MILVUS_METRIC_TYPE",
    )

    # LLM 服务配置。
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
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
    llm_model_name: str = Field(default="gpt-4o-mini", alias="LLM_MODEL_NAME")
    embedding_model_name: str = Field(
        # `BAAI/bge-m3` 支持中文、多语种和较长文本检索，
        # 对 FAQ、SOP、产品文档、工单经验库这类典型企业知识库更友好。
        default="BAAI/bge-m3",
        alias="EMBEDDING_MODEL_NAME",
    )
    reranker_model_name: str = Field(
        # reranker 负责在召回候选上做最后一轮精排，
        # 它对“最终给 LLM 喂哪些 chunk”影响非常大。
        # 这里同样优先采用中文检索生态里更常见的 BGE reranker 默认值。
        default="BAAI/bge-reranker-v2-m3",
        alias="RERANKER_MODEL_NAME",
    )

    # 检索链路的召回 / 融合 / 重排参数。
    bm25_top_k: int = Field(default=20, alias="BM25_TOP_K")
    dense_top_k: int = Field(default=20, alias="DENSE_TOP_K")
    hybrid_top_k: int = Field(default=30, alias="HYBRID_TOP_K")
    rerank_top_n: int = Field(default=8, alias="RERANK_TOP_N")
    fusion_strategy: Literal["rrf", "weighted"] = Field(
        default="rrf", alias="FUSION_STRATEGY"
    )
    fusion_sparse_weight: float = Field(default=0.5, alias="FUSION_SPARSE_WEIGHT")
    faq_bm25_threshold: float = Field(default=0.85, alias="FAQ_BM25_THRESHOLD")
    faq_top_k: int = Field(default=3, alias="FAQ_TOP_K")
    answer_cache_ttl_sec: int = Field(default=86400, alias="ANSWER_CACHE_TTL_SEC")
    faq_seed_path: str = Field(default="./data/mock_corpus/faq_seed.csv", alias="FAQ_SEED_PATH")

    # 拒答门槛：
    # - `min_retrieval_score` 用在召回层的最低可用阈值
    # - `min_rerank_score` 用在生成前的最终相关性阈值
    # - `refusal_confidence_threshold` 用在后处理阶段判断是否该拒答
    min_retrieval_score: float = Field(default=0.15, alias="MIN_RETRIEVAL_SCORE")
    min_rerank_score: float = Field(default=-5.0, alias="MIN_RERANK_SCORE")
    refusal_confidence_threshold: float = Field(
        default=0.35, alias="REFUSAL_CONFIDENCE_THRESHOLD"
    )

    # 可观测性配置：如果提供 OTLP 端点，就会把 trace 导出到收集器。
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    otel_service_name: str = Field(
        default="enterprise-rag-api", alias="OTEL_SERVICE_NAME"
    )

    # 评测配置：样例数据集和报告输出目录。
    eval_output_dir: str = Field(default="./data/eval_reports", alias="EVAL_OUTPUT_DIR")
    eval_dataset_path: str = Field(
        default="./core/evaluation/datasets/sample_eval.jsonl", alias="EVAL_DATASET_PATH"
    )

    # 前端开发模式与跨域部署都依赖这组来源配置。
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ORIGINS",
    )


@lru_cache
def get_settings() -> Settings:
    """返回进程级单例配置。

    `lru_cache` 的作用：
    - 避免每次调用都重复解析环境变量。
    - 保证全进程拿到的是同一份配置对象，便于调试和测试。
    """

    return Settings()
