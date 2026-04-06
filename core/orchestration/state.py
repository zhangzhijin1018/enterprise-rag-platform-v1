"""LangGraph 状态定义模块。统一描述图节点之间传递的数据结构。"""

from __future__ import annotations

from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    question: str
    conversation_id: str | None
    top_k_sparse: int | None
    top_k_dense: int | None
    rerank_top_n: int | None

    query_type: str
    rewritten_query: str
    multi_queries: list[str]

    sparse_hits: list[dict[str, Any]]
    dense_hits: list[dict[str, Any]]
    fused_hits: list[dict[str, Any]]
    reranked_hits: list[dict[str, Any]]

    answer: str
    confidence: float
    reasoning_summary: str
    citations: list[dict[str, Any]]
    refusal: bool
    refusal_reason: str

    grounding_ok: bool
    errors: list[str]
