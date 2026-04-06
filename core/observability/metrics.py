"""指标定义模块。集中声明 Prometheus 指标，并提供 /metrics 响应辅助函数。"""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

metrics_registry = CollectorRegistry()

REQUEST_LATENCY = Histogram(
    "erp_http_request_duration_seconds",
    "HTTP request latency",
    ["route", "method"],
    registry=metrics_registry,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)

RETRIEVAL_LATENCY = Histogram(
    "erp_retrieval_duration_seconds",
    "Retrieval stage latency",
    ["stage"],
    registry=metrics_registry,
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2),
)

RERANK_LATENCY = Histogram(
    "erp_rerank_duration_seconds",
    "Reranker latency",
    registry=metrics_registry,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

TOKENS_USED = Counter(
    "erp_llm_tokens_total",
    "Approximate LLM tokens (prompt+completion)",
    ["model", "kind"],
    registry=metrics_registry,
)

EMPTY_RETRIEVAL = Counter(
    "erp_empty_retrieval_total",
    "Queries with zero fused hits",
    registry=metrics_registry,
)

CITATION_COVERAGE = Histogram(
    "erp_citation_coverage_ratio",
    "Ratio of citations used vs retrieved chunks",
    registry=metrics_registry,
    buckets=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

FAITHFULNESS_SCORE = Gauge(
    "erp_ragas_faithfulness_avg",
    "Rolling / last eval faithfulness (set by eval job)",
    registry=metrics_registry,
)


def metrics_response() -> tuple[bytes, str]:
    data = generate_latest(metrics_registry)
    return data, CONTENT_TYPE_LATEST
