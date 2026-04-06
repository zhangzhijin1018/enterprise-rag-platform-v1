"""重排节点模块。负责把融合后的候选文档再次排序，得到更可靠的最终上下文。"""

from __future__ import annotations

import time

from core.observability.metrics import RERANK_LATENCY
from core.orchestration.state import RAGState
from core.retrieval.schemas import RetrievedChunk
from core.services.runtime import RAGRuntime


async def rerank_docs_node(state: RAGState, runtime: RAGRuntime) -> RAGState:
    rq = state.get("rewritten_query") or state.get("question") or ""
    fused_raw = state.get("fused_hits") or []
    fused = [RetrievedChunk.model_validate(x) for x in fused_raw]
    if not fused:
        return {"reranked_hits": []}
    topn = state.get("rerank_top_n") or runtime.settings.rerank_top_n
    t0 = time.perf_counter()
    reranked = runtime.reranker.rerank(rq, fused, top_n=topn)
    RERANK_LATENCY.observe(time.perf_counter() - t0)
    return {"reranked_hits": [h.model_dump(mode="json") for h in reranked]}
