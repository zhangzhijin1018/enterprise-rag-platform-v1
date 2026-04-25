"""重排节点模块。负责把融合后的候选文档再次排序，得到更可靠的最终上下文。"""

from __future__ import annotations

import time

from core.observability.metrics import RERANK_LATENCY
from core.orchestration.state import RAGState
from core.retrieval.governance import apply_governance_ranking, detect_document_conflicts
from core.retrieval.schemas import RetrievedChunk
from core.services.runtime import RAGRuntime


async def rerank_docs_node(state: RAGState, runtime: RAGRuntime) -> RAGState:
    rq = state.get("rewritten_query") or state.get("question") or ""
    fused_raw = state.get("fused_hits") or []
    fused = [RetrievedChunk.model_validate(x) for x in fused_raw]
    if not fused:
        return {
            "reranked_hits": [],
            "conflict_detected": False,
            "conflict_summary": "",
        }
    topn = state.get("rerank_top_n") or runtime.settings.rerank_top_n
    candidate_limit = min(
        len(fused),
        max(
            topn,
            min(
                int(getattr(runtime.settings, "rerank_candidate_max", len(fused))),
                topn * int(getattr(runtime.settings, "rerank_candidate_multiplier", 1)),
            ),
        ),
    )
    rerank_candidates = fused[:candidate_limit]
    t0 = time.perf_counter()
    reranked = runtime.reranker.rerank(rq, rerank_candidates, top_n=topn)
    RERANK_LATENCY.observe(time.perf_counter() - t0)
    reranked = apply_governance_ranking(reranked, runtime.settings)
    conflict_detected, conflict_summary = detect_document_conflicts(reranked, runtime.settings)
    answer_mode = "grounded_answer_with_conflict" if conflict_detected else "grounded_answer"
    reranked_dump = [h.model_dump(mode="json") for h in reranked]
    for item in reranked_dump:
        trace = dict(item.get("trace") or {})
        trace["rerank_candidate_limit"] = candidate_limit
        trace["rerank_input_size"] = len(fused)
        item["trace"] = trace
    return {
        "reranked_hits": reranked_dump,
        "conflict_detected": conflict_detected,
        "conflict_summary": conflict_summary,
        "answer_mode": answer_mode,
    }
