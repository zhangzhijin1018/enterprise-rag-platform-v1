"""仅检索流水线模块。

流式回答场景下，通常希望尽快把“检索到的片段和引用”先返回给前端，
因此这里提供一个不包含最终生成阶段的精简链路。
"""

from __future__ import annotations

from core.orchestration.fast_path import try_fast_path_answer
from core.orchestration.fusion_gate import fusion_results_actionable
from core.orchestration.nodes.analyze_query import analyze_query_node
from core.orchestration.nodes.clarify_query import (
    clarify_query_node,
    request_clarification_node,
)
from core.orchestration.nodes.resolve_context import resolve_context_node
from core.orchestration.nodes.retrieve_docs import retrieve_docs_node
from core.orchestration.nodes.rerank_docs import rerank_docs_node
from core.orchestration.nodes.rewrite_query import rewrite_query_node
from core.orchestration.policies.fallback import empty_retrieval_refusal
from core.orchestration.state import RAGState
from core.services.runtime import RAGRuntime


def _merge(base: RAGState, update: RAGState) -> RAGState:
    """合并节点返回的状态增量。

    LangGraph 节点通常只返回“本节点新增或更新的字段”，
    所以这里用一个轻量 merge 把增量写回当前状态。
    """

    out = dict(base)
    out.update(update)
    return out  # type: ignore[return-value]


def _empty_retrieval_response(state: RAGState) -> RAGState:
    """在 retrieval-only 链路里保留上游已确定的拒答语义。

    retrieval-only 模式虽然不生成最终答案，但仍需要把“为什么拒答”说明清楚，
    否则前端只能看到空结果，不知道是：
    - 真没检索到
    - 分数太低
    - 上游已经决定拒答
    """

    base = empty_retrieval_refusal(state)
    base["answer_mode"] = "refusal"
    if state.get("refusal_reason"):
        base["refusal_reason"] = state["refusal_reason"]
    if state.get("answer"):
        base["answer"] = state["answer"]
    if state.get("reasoning_summary"):
        base["reasoning_summary"] = state["reasoning_summary"]
    return base


async def run_retrieval_only(
    runtime: RAGRuntime,
    *,
    question: str,
    conversation_id: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    top_k_sparse: int | None = None,
    top_k_dense: int | None = None,
    rerank_top_n: int | None = None,
    user_context: dict[str, object] | None = None,
    access_filters: dict[str, object] | None = None,
    risk_state: dict[str, object] | None = None,
    audit_id: str | None = None,
    disable_fast_path: bool = False,
) -> RAGState:
    """执行“分析 -> 澄清 -> 上下文补全 -> 规划 -> 检索 -> 重排”的精简链路。

    这个函数主要服务于：
    - 流式回答前的“先给证据片段”
    - 只想看检索效果、不想跑最终生成时的调试
    - 某些前端想把“检索”和“生成”拆成两段展示

    和完整 RAG 图相比，它有意停在 rerank 之后，不进入 generate。
    """

    fast = None
    if not disable_fast_path:
        fast = await try_fast_path_answer(runtime, question)
    if fast is not None:
        # 即使是 fast path，也统一返回 `RAGState` 形状，
        # 这样调用方不需要为 retrieval-only 再写一套分支解析逻辑。
        return {
            **fast,
            "conversation_id": conversation_id,
            "history_messages": history_messages or [],
            "user_context": user_context or {},
            "access_filters": access_filters or {},
            **(risk_state or {}),
            "audit_id": audit_id or "",
            "errors": [],
        }

    state: RAGState = {
        "question": question,
        "conversation_id": conversation_id,
        "history_messages": history_messages or [],
        "top_k_sparse": top_k_sparse,
        "top_k_dense": top_k_dense,
        "rerank_top_n": rerank_top_n,
        "user_context": user_context or {},
        "access_filters": access_filters or {},
        **(risk_state or {}),
        "audit_id": audit_id or "",
        "errors": [],
    }
    state = _merge(state, await analyze_query_node(state))
    state = _merge(state, await clarify_query_node(state, runtime))
    if state.get("need_clarify"):
        # 如果当前问题信息明显不足，就在进入检索前先停下来，
        # 避免“带着模糊问题去搜，最后搜出一堆不稳的候选”。
        return _merge(state, await request_clarification_node(state))
    state = _merge(state, await resolve_context_node(state, runtime))
    state = _merge(state, await rewrite_query_node(state, runtime))
    state = _merge(state, await retrieve_docs_node(state, runtime))
    fused = state.get("fused_hits") or []
    if not fusion_results_actionable(runtime.settings, fused):
        # 这里不是简单判断“有没有召回”，而是判断“这些候选是否值得继续”。
        # 不值得时直接返回 refusal 语义，比盲目继续 rerank 更稳。
        return _merge(state, _empty_retrieval_response(state))
    state = _merge(state, await rerank_docs_node(state, runtime))
    return state
