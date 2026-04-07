"""查询改写节点模块。

这个节点虽然保留了历史名称 `rewrite_query_node`，
但在第二轮增强后，它已经不再只做“单句改写”，而是在做轻量查询规划：

- 生成更稳的主 query
- 生成若干子问题 query
- 生成关键词 query
- 可选生成 HyDE query

保留旧名称的原因只是为了减少对现有项目结构的破坏，
并不代表它的职责仍然只停留在 rewrite。
"""

from __future__ import annotations

from core.orchestration.query_expansion import build_query_plan
from core.orchestration.state import RAGState
from core.services.runtime import RAGRuntime


async def rewrite_query_node(state: RAGState, runtime: RAGRuntime) -> RAGState:
    """执行查询规划。

    这个节点保留了历史名称 `rewrite_query_node`，
    但职责已经从“只产出一个改写 query”升级为“产出一份检索计划”：

    - `rewritten_query`
    - `multi_queries`
    - `keyword_queries`
    - `hyde_query`

    这样做的原因是：真实企业问答里，单一 query 往往不足以兼顾术语精确匹配、
    语义相似检索和复杂问题拆分。
    """

    q = state.get("question") or ""
    plan = await build_query_plan(
        runtime,
        question=q,
        query_type=state.get("query_type", "general"),
    )
    return {
        "rewritten_query": plan.rewritten_query[:512],
        "multi_queries": plan.multi_queries,
        "keyword_queries": plan.keyword_queries,
        "hyde_query": plan.hyde_query[:1024],
        "query_plan_summary": plan.planning_summary,
    }
