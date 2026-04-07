"""LangGraph 状态定义模块。统一描述图节点之间传递的数据结构。"""

from __future__ import annotations

from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    # 原始输入与运行时控制参数。
    question: str
    conversation_id: str | None
    top_k_sparse: int | None
    top_k_dense: int | None
    rerank_top_n: int | None

    # 查询分析与澄清阶段产物。
    query_type: str
    need_clarify: bool
    missing_slots: list[str]
    clarify_question: str
    clarify_reason: str

    # 查询规划阶段产物。
    rewritten_query: str
    keyword_queries: list[str]
    multi_queries: list[str]
    hyde_query: str
    query_plan_summary: str

    # 检索与重排阶段产物。
    sparse_hits: list[dict[str, Any]]
    dense_hits: list[dict[str, Any]]
    fused_hits: list[dict[str, Any]]
    reranked_hits: list[dict[str, Any]]

    # 生成与最终校验阶段产物。
    answer: str
    confidence: float
    reasoning_summary: str
    citations: list[dict[str, Any]]
    fast_path_source: str
    refusal: bool
    refusal_reason: str

    grounding_ok: bool
    errors: list[str]
