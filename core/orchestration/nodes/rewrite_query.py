"""查询规划节点模块。

这个节点保留了历史名称 `rewrite_query_node`，
但职责已经升级为“构建检索计划”，而不是只做单句改写。
"""

from __future__ import annotations

from core.orchestration.query_expansion import build_query_plan
from core.orchestration.state import RAGState
from core.services.runtime import RAGRuntime


async def rewrite_query_node(state: RAGState, runtime: RAGRuntime) -> RAGState:
    """执行查询规划，生成多路 query routes 与 structured filters。"""

    question = state.get("question") or ""
    plan = await build_query_plan(
        runtime,
        question=question,
        resolved_query=state.get("resolved_query") or "",
        strategy_signals=state.get("strategy_signals") or {},
    )
    return {
        "resolved_query": plan.resolved_query[:512],
        "rewritten_query": plan.rewritten_query[:512],
        "multi_queries": plan.multi_queries,
        "keyword_queries": plan.keyword_queries,
        "hyde_query": plan.hyde_query[:1024],
        "structured_filters": plan.structured_filters,
        "query_plan_summary": plan.planning_summary,
    }
