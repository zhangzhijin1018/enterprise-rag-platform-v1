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

    # LLM 服务配置。
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )

    # 模型选择：
    # - `llm_model_name` 用于回答生成 / 查询改写
    # - `embedding_model_name` 用于把文本映射成向量
    # - `reranker_model_name` 用于二次精排
    llm_model_name: str = Field(default="gpt-4o-mini", alias="LLM_MODEL_NAME")
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_NAME"
    )
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL_NAME"
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
