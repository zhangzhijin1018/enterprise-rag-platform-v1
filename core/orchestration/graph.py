"""RAG 状态图定义模块。

当前问答流程显式建模为：

`analyze_signals -> clarify_gate -> resolve_context -> build_query_plan -> retrieve -> rerank -> generate -> validate`

关键变化：
- 不再依赖固定业务三分类驱动后续流程
- `clarify` 是检索规划前的前置闸门，而不是 query route
- 新增 `resolved_query` 路线，专门处理多轮上下文承接
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from core.orchestration.fast_path import try_fast_path_answer
from core.orchestration.fusion_gate import fusion_results_actionable
from core.orchestration.nodes.analyze_query import analyze_query_node
from core.orchestration.nodes.clarify_query import (
    clarify_query_node,
    request_clarification_node,
)
from core.orchestration.nodes.generate_answer import generate_answer_node
from core.orchestration.nodes.resolve_context import resolve_context_node
from core.orchestration.nodes.retrieve_docs import retrieve_docs_node
from core.orchestration.nodes.rerank_docs import rerank_docs_node
from core.orchestration.nodes.rewrite_query import rewrite_query_node
from core.orchestration.nodes.validate_grounding import validate_grounding_node
from core.orchestration.policies.fallback import empty_retrieval_refusal
from core.orchestration.state import RAGState
from core.services.runtime import RAGRuntime


def _empty_retrieval_response(state: RAGState) -> RAGState:
    """在空召回时保留上游已确定的拒答语义。

    这样可以确保：
    - 空召回不会被误解释成系统异常
    - 上游已经写入的 refusal_reason / reasoning_summary 不会丢
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


def build_rag_graph(runtime: RAGRuntime) -> StateGraph:
    """构建 LangGraph 状态图。

    这里把主流程显式拆成节点，而不是在一个大函数里串所有步骤，
    目的是让：
    - 每个阶段职责清晰
    - 中间状态可观察
    - 后续插入企业安全、风控、治理节点更容易

    你可以把这个图理解成“可视化的主流程骨架”：
    节点负责做事，边负责决定下一步走向。
    """

    graph = StateGraph(RAGState)

    async def analyze_signals(s: RAGState) -> RAGState:
        return await analyze_query_node(s, runtime)

    async def clarify_gate(s: RAGState) -> RAGState:
        return await clarify_query_node(s, runtime)

    async def ask_clarify(s: RAGState) -> RAGState:
        return await request_clarification_node(s)

    async def resolve_context(s: RAGState) -> RAGState:
        return await resolve_context_node(s, runtime)

    async def build_query_plan(s: RAGState) -> RAGState:
        return await rewrite_query_node(s, runtime)

    async def retrieve(s: RAGState) -> RAGState:
        return await retrieve_docs_node(s, runtime)

    async def rerank(s: RAGState) -> RAGState:
        return await rerank_docs_node(s, runtime)

    async def generate(s: RAGState) -> RAGState:
        return await generate_answer_node(s, runtime)

    async def validate(s: RAGState) -> RAGState:
        return await validate_grounding_node(s)

    async def refuse_empty(s: RAGState) -> RAGState:
        # 即使是空召回拒答，也把 `reranked_hits` 明确设为空，
        # 避免调用方误以为“只是没展开字段”。
        base = _empty_retrieval_response(s)
        return {
            **base,
            "reranked_hits": [],
        }

    graph.add_node("analyze_signals", analyze_signals)
    graph.add_node("clarify_gate", clarify_gate)
    graph.add_node("ask_clarify", ask_clarify)
    graph.add_node("resolve_context", resolve_context)
    graph.add_node("build_query_plan", build_query_plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_node("refuse_empty", refuse_empty)

    # 图入口固定从查询理解开始。
    graph.add_edge(START, "analyze_signals")
    graph.add_edge("analyze_signals", "clarify_gate")

    def route_clarify(state: RAGState) -> str:
        # `clarify_gate` 不是最终回答节点，而是一个前置闸门：
        # 只有在槽位明显缺失、问题太短或指代不明时才会停下来澄清。
        if state.get("need_clarify"):
            return "ask"
        return "continue"

    graph.add_conditional_edges(
        "clarify_gate",
        route_clarify,
        {"ask": "ask_clarify", "continue": "resolve_context"},
    )
    graph.add_edge("resolve_context", "build_query_plan")
    graph.add_edge("build_query_plan", "retrieve")

    def route_retrieve(state: RAGState) -> str:
        # retrieve 后不是只看“有没有召回”，而是看“这些候选是否值得继续走 rerank / generate”。
        fused = state.get("fused_hits") or []
        if not fusion_results_actionable(runtime.settings, fused):
            return "empty"
        return "ok"

    graph.add_conditional_edges(
        "retrieve",
        route_retrieve,
        {"empty": "refuse_empty", "ok": "rerank"},
    )
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "validate")
    # `ask_clarify` 和 `refuse_empty` 也统一进入 validate，
    # 是为了让“最终输出收口逻辑”保持一致：
    # 不管是正常回答、澄清、还是拒答，最终都从同一个出口结束。
    graph.add_edge("ask_clarify", "validate")
    graph.add_edge("refuse_empty", "validate")
    graph.add_edge("validate", END)

    return graph


async def run_rag_async(
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
    """执行完整 RAG 图。

    这里额外保留了一个 fast path：
    - Redis 热点答案
    - MySQL FAQ

    只有 fast path 没命中时，才进入完整 LangGraph。

    这样做是一个典型的工程折中：
    - 高频、标准化问题尽量快返回
    - 复杂、开放式问题再走完整 RAG 图
    """

    fast = None
    if not disable_fast_path:
        fast = await try_fast_path_answer(runtime, question)
    if fast is not None:
        # fast path 也统一收敛成 `RAGState`，这样 API 层和前端都不需要区分快慢路径。
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
    app = runtime.get_compiled_graph()
    # 真正图执行时，入口状态尽量只放“问题 + 请求参数 + 安全上下文”，
    # 中间产物由各节点逐步补全。
    return await app.ainvoke(
        {
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
    )
