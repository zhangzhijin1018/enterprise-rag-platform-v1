"""RAG 状态图定义模块。

这里把问答流程显式建模成一个状态机：
`analyze -> clarify -> rewrite -> retrieve -> rerank -> generate -> validate`

第二轮增强新增了 `clarify` 分支：
- 如果问题缺少关键信息，就先追问用户；
- 如果问题已经足够明确，再进入多路检索链路。

空召回时仍然走 `refuse_empty -> validate` 分支。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from core.orchestration.nodes.analyze_query import analyze_query_node
from core.orchestration.nodes.clarify_query import (
    clarify_query_node,
    request_clarification_node,
)
from core.orchestration.nodes.generate_answer import generate_answer_node
from core.orchestration.nodes.retrieve_docs import retrieve_docs_node
from core.orchestration.nodes.rerank_docs import rerank_docs_node
from core.orchestration.nodes.rewrite_query import rewrite_query_node
from core.orchestration.nodes.validate_grounding import validate_grounding_node
from core.orchestration.fusion_gate import fusion_results_actionable
from core.orchestration.fast_path import try_fast_path_answer
from core.orchestration.policies.fallback import empty_retrieval_refusal
from core.orchestration.state import RAGState
from core.services.runtime import RAGRuntime


def build_rag_graph(runtime: RAGRuntime) -> StateGraph:
    """构建 LangGraph 状态图。"""

    graph = StateGraph(RAGState)

    # 这些包装函数的目的，是把 runtime 依赖注入到节点里，同时保持节点签名清晰。
    async def analyze(s: RAGState) -> RAGState:
        return await analyze_query_node(s)

    async def clarify(s: RAGState) -> RAGState:
        return await clarify_query_node(s, runtime)

    async def ask_clarify(s: RAGState) -> RAGState:
        return await request_clarification_node(s)

    async def rewrite(s: RAGState) -> RAGState:
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
        # 空召回时不再进入重排和生成，而是直接构造拒答结果。
        base = empty_retrieval_refusal(s)
        return {
            **base,
            "reranked_hits": [],
        }

    # 注册节点。
    graph.add_node("analyze", analyze)
    graph.add_node("clarify", clarify)
    graph.add_node("ask_clarify", ask_clarify)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_node("refuse_empty", refuse_empty)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "clarify")

    def route_clarify(state: RAGState) -> str:
        """决定是先向用户追问，还是继续进入检索链路。"""

        if state.get("need_clarify"):
            return "ask"
        return "continue"

    graph.add_conditional_edges(
        "clarify",
        route_clarify,
        {"ask": "ask_clarify", "continue": "rewrite"},
    )
    graph.add_edge("rewrite", "retrieve")

    def route_retrieve(state: RAGState) -> str:
        """决定召回后走哪条分支。"""

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
    graph.add_edge("ask_clarify", "validate")
    graph.add_edge("refuse_empty", "validate")
    graph.add_edge("validate", END)

    return graph


async def run_rag_async(
    runtime: RAGRuntime,
    *,
    question: str,
    conversation_id: str | None = None,
    top_k_sparse: int | None = None,
    top_k_dense: int | None = None,
    rerank_top_n: int | None = None,
) -> RAGState:
    """执行完整 RAG 图。"""

    # 先走 Redis / MySQL FAQ 快速通道。
    # 只有两层都没命中，才进入完整 LangGraph RAG 链路。
    fast = await try_fast_path_answer(runtime, question)
    if fast is not None:
        return {
            **fast,
            "conversation_id": conversation_id,
            "errors": [],
        }
    app = runtime.get_compiled_graph()
    return await app.ainvoke(
        {
            "question": question,
            "conversation_id": conversation_id,
            "top_k_sparse": top_k_sparse,
            "top_k_dense": top_k_dense,
            "rerank_top_n": rerank_top_n,
            "errors": [],
        }
    )
