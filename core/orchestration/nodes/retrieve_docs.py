"""检索节点模块。

该节点会同时执行：
1. BM25 稀疏召回
2. 向量稠密召回
3. 两路结果融合
"""

from __future__ import annotations

import time

from core.observability.metrics import EMPTY_RETRIEVAL, RETRIEVAL_LATENCY
from core.orchestration.state import RAGState
from core.retrieval.schemas import RetrievedChunk
from core.services.runtime import RAGRuntime


def _dump(hits: list[RetrievedChunk]) -> list[dict]:
    """把检索结果转成可安全写入 LangGraph state 的 JSON 结构。"""

    return [h.model_dump(mode="json") for h in hits]


async def retrieve_docs_node(state: RAGState, runtime: RAGRuntime) -> RAGState:
    """执行混合检索节点。"""

    rq = state.get("rewritten_query") or state.get("question") or ""
    t0 = time.perf_counter()
    sk = state.get("top_k_sparse") or runtime.settings.bm25_top_k
    dk = state.get("top_k_dense") or runtime.settings.dense_top_k
    # 稀疏检索更擅长精确术语，如错误码、命令、产品名。
    sparse = runtime.sparse.search(rq, top_k=sk)
    RETRIEVAL_LATENCY.labels(stage="sparse").observe(time.perf_counter() - t0)
    t1 = time.perf_counter()
    # 稠密检索更擅长语义近似匹配。
    dense = runtime.dense.search(rq, top_k=dk)
    RETRIEVAL_LATENCY.labels(stage="dense").observe(time.perf_counter() - t1)
    t2 = time.perf_counter()
    # 融合阶段把两路结果统一排序，兼顾召回率与稳定性。
    fused = runtime.fusion.fuse(sparse, dense)
    RETRIEVAL_LATENCY.labels(stage="fusion").observe(time.perf_counter() - t2)
    if not fused:
        EMPTY_RETRIEVAL.inc()
    return {
        "sparse_hits": _dump(sparse),
        "dense_hits": _dump(dense),
        "fused_hits": _dump(fused),
    }
