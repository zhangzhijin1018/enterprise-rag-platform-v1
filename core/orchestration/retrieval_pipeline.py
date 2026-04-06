"""仅检索流水线模块。

流式回答场景下，通常希望尽快把“检索到的片段和引用”先返回给前端，
因此这里提供一个不包含最终生成阶段的精简链路。
"""

from __future__ import annotations

from core.orchestration.nodes.analyze_query import analyze_query_node
from core.orchestration.nodes.retrieve_docs import retrieve_docs_node
from core.orchestration.nodes.rerank_docs import rerank_docs_node
from core.orchestration.nodes.rewrite_query import rewrite_query_node
from core.orchestration.fusion_gate import fusion_results_actionable
from core.orchestration.policies.fallback import empty_retrieval_refusal
from core.orchestration.state import RAGState
from core.services.runtime import RAGRuntime


def _merge(base: RAGState, update: RAGState) -> RAGState:
    """合并节点返回的状态增量。"""

    out = dict(base)
    out.update(update)
    return out  # type: ignore[return-value]


async def run_retrieval_only(
    runtime: RAGRuntime,
    *,
    question: str,
    conversation_id: str | None = None,
    top_k_sparse: int | None = None,
    top_k_dense: int | None = None,
    rerank_top_n: int | None = None,
) -> RAGState:
    """执行“分析 -> 改写 -> 检索 -> 重排”的精简链路。"""

    state: RAGState = {
        "question": question,
        "conversation_id": conversation_id,
        "top_k_sparse": top_k_sparse,
        "top_k_dense": top_k_dense,
        "rerank_top_n": rerank_top_n,
        "errors": [],
    }
    # 这里显式顺序执行，而不是通过完整 LangGraph compile，目的是让流式场景更轻量。
    state = _merge(state, await analyze_query_node(state))
    state = _merge(state, await rewrite_query_node(state, runtime))
    state = _merge(state, await retrieve_docs_node(state, runtime))
    fused = state.get("fused_hits") or []
    if not fusion_results_actionable(runtime.settings, fused):
        return _merge(state, empty_retrieval_refusal(state))
    state = _merge(state, await rerank_docs_node(state, runtime))
    return state
