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
from core.models.document import TextChunk
from core.retrieval.schemas import RetrievedChunk
from core.services.runtime import RAGRuntime


def _dump(hits: list[RetrievedChunk]) -> list[dict]:
    """把检索结果转成可安全写入 LangGraph state 的 JSON 结构。"""

    return [h.model_dump(mode="json") for h in hits]


def _dedupe_queries(queries: list[str]) -> list[str]:
    """按顺序去重 query 列表。"""

    seen: set[str] = set()
    out: list[str] = []
    for query in queries:
        text = query.strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _annotate_hits(
    hits: list[RetrievedChunk],
    *,
    route_name: str,
    route_query: str,
    route_kind: str,
) -> list[RetrievedChunk]:
    """给检索结果补充 query 路线 trace。"""

    out: list[RetrievedChunk] = []
    for hit in hits:
        trace = dict(hit.trace)
        trace["query_route"] = route_name
        trace["query_route_kind"] = route_kind
        trace["query_text"] = route_query[:256]
        out.append(hit.model_copy(update={"trace": trace}))
    return out


def _merge_hits_by_chunk(hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """把多路 query 的命中结果按 chunk_id 聚合。"""

    merged: dict[str, RetrievedChunk] = {}
    route_map: dict[str, list[str]] = {}
    kind_map: dict[str, list[str]] = {}
    for hit in hits:
        cid = hit.chunk_id
        route = str(hit.trace.get("query_route") or "")
        kind = str(hit.trace.get("query_route_kind") or "")
        route_map.setdefault(cid, [])
        kind_map.setdefault(cid, [])
        if route and route not in route_map[cid]:
            route_map[cid].append(route)
        if kind and kind not in kind_map[cid]:
            kind_map[cid].append(kind)

        existing = merged.get(cid)
        if existing is None or hit.score > existing.score:
            merged[cid] = hit

    out: list[RetrievedChunk] = []
    for cid, base in merged.items():
        trace = dict(base.trace)
        trace["matched_routes"] = route_map.get(cid, [])
        trace["matched_route_kinds"] = kind_map.get(cid, [])
        trace["matched_route_count"] = len(route_map.get(cid, []))
        out.append(base.model_copy(update={"trace": trace}))
    return out


def _resolve_parent_chunk(runtime: RAGRuntime, hit: RetrievedChunk) -> TextChunk | None:
    """根据 child 命中结果找到它对应的 parent chunk。

    解析规则：

    1. 如果当前命中的 chunk 本身就是 parent，直接返回自己；
    2. 如果 metadata 里带了 `parent_chunk_id`，优先按这个 id 回查；
    3. 如果查不到 parent，就返回 `None`，调用方再回退使用 child 自身。
    """

    if hit.metadata.is_parent:
        return runtime.store.get_chunk_by_id(hit.chunk_id)
    parent_chunk_id = hit.metadata.parent_chunk_id
    if not parent_chunk_id:
        return None
    return runtime.store.get_chunk_by_id(parent_chunk_id)


def _expand_hits_to_parent_chunks(
    runtime: RAGRuntime,
    hits: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """把 child 命中结果回扩成 parent 命中结果。

    这是 parent-child 检索的核心步骤之一。

    为什么需要回扩：

    - child chunk 适合做召回，因为更短、更聚焦；
    - 但 rerank 和 generation 更需要完整上下文；
    - 所以检索命中 child 后，要把它映射回 parent，再交给后续链路。

    当前聚合策略：

    - 以 `parent_chunk_id` 为聚合键；
    - parent 分数取其所有命中 child 分数的最大值；
    - trace 里保留命中的 child ids、命中路线数量等调试信息。

    为什么先用 `max(child_score)`：

    - 这是最稳、最容易解释的默认策略；
    - 如果 parent 下有一个 child 和 query 非常贴近，就说明这个 parent 至少值得进入候选；
    - 相比平均分，`max` 不容易被 parent 下其他无关 child 稀释掉。
    """

    merged: dict[str, RetrievedChunk] = {}
    child_ids_map: dict[str, list[str]] = {}
    routes_map: dict[str, list[str]] = {}
    route_kind_map: dict[str, list[str]] = {}
    for hit in hits:
        parent_chunk = _resolve_parent_chunk(runtime, hit)
        if parent_chunk is None:
            # 回查失败时回退到 child 自身，确保链路可用性优先。
            parent_chunk = TextChunk(content=hit.content, metadata=hit.metadata)
        parent_id = parent_chunk.metadata.chunk_id

        child_ids_map.setdefault(parent_id, [])
        routes_map.setdefault(parent_id, [])
        route_kind_map.setdefault(parent_id, [])
        if hit.chunk_id not in child_ids_map[parent_id]:
            child_ids_map[parent_id].append(hit.chunk_id)
        for route in hit.trace.get("matched_routes") or []:
            if isinstance(route, str) and route not in routes_map[parent_id]:
                routes_map[parent_id].append(route)
        for route_kind in hit.trace.get("matched_route_kinds") or []:
            if isinstance(route_kind, str) and route_kind not in route_kind_map[parent_id]:
                route_kind_map[parent_id].append(route_kind)

        existing = merged.get(parent_id)
        base_trace = dict(hit.trace)
        base_trace["expanded_to_parent"] = parent_id != hit.chunk_id
        if existing is None or hit.score > existing.score:
            merged[parent_id] = RetrievedChunk(
                chunk_id=parent_id,
                score=hit.score,
                content=parent_chunk.content,
                metadata=parent_chunk.metadata,
                trace=base_trace,
            )

    out: list[RetrievedChunk] = []
    for parent_id, parent_hit in merged.items():
        trace = dict(parent_hit.trace)
        trace["matched_child_chunk_ids"] = child_ids_map.get(parent_id, [])
        trace["matched_child_count"] = len(child_ids_map.get(parent_id, []))
        trace["matched_routes"] = routes_map.get(parent_id, [])
        trace["matched_route_kinds"] = route_kind_map.get(parent_id, [])
        trace["matched_route_count"] = len(routes_map.get(parent_id, []))
        out.append(parent_hit.model_copy(update={"trace": trace}))
    return out


async def retrieve_docs_node(state: RAGState, runtime: RAGRuntime) -> RAGState:
    """执行多路混合检索节点。"""

    sk = state.get("top_k_sparse") or runtime.settings.bm25_top_k
    dk = state.get("top_k_dense") or runtime.settings.dense_top_k
    question = state.get("question") or ""
    rewritten = state.get("rewritten_query") or question
    multi_queries = state.get("multi_queries") or []
    keyword_queries = state.get("keyword_queries") or []
    hyde_query = state.get("hyde_query") or ""

    # 不同 query 路线适合不同检索器：
    # - original / rewrite / sub_query：同时走 sparse + dense；
    # - keyword：主要走 sparse；
    # - hyde：主要走 dense。
    query_routes: list[tuple[str, str, str, bool, bool]] = [("original", question, "original", True, True)]
    if rewritten.strip():
        query_routes.append(("rewrite", rewritten, "rewrite", True, True))
    for idx, sub_query in enumerate(_dedupe_queries(multi_queries), start=1):
        query_routes.append((f"sub_query_{idx}", sub_query, "sub_query", True, True))
    for idx, keyword_query in enumerate(_dedupe_queries(keyword_queries), start=1):
        query_routes.append((f"keyword_{idx}", keyword_query, "keyword", True, False))
    if hyde_query.strip():
        query_routes.append(("hyde", hyde_query, "hyde", False, True))

    sparse_all: list[RetrievedChunk] = []
    dense_all: list[RetrievedChunk] = []
    sparse_elapsed = 0.0
    dense_elapsed = 0.0
    for route_name, route_query, route_kind, use_sparse, use_dense in query_routes:
        if use_sparse:
            t_sparse = time.perf_counter()
            sparse_hits = runtime.sparse.search(route_query, top_k=sk)
            sparse_elapsed += time.perf_counter() - t_sparse
            sparse_all.extend(
                _annotate_hits(
                    sparse_hits,
                    route_name=route_name,
                    route_query=route_query,
                    route_kind=route_kind,
                )
            )
        if use_dense:
            t_dense = time.perf_counter()
            dense_hits = runtime.dense.search(route_query, top_k=dk)
            dense_elapsed += time.perf_counter() - t_dense
            dense_all.extend(
                _annotate_hits(
                    dense_hits,
                    route_name=route_name,
                    route_query=route_query,
                    route_kind=route_kind,
                )
            )
    RETRIEVAL_LATENCY.labels(stage="sparse").observe(sparse_elapsed)
    RETRIEVAL_LATENCY.labels(stage="dense").observe(dense_elapsed)

    # 第一步：先在 child 命中层面做去重，把“同一 child 被多条 query 命中”的情况收敛掉。
    sparse_child_hits = _merge_hits_by_chunk(sparse_all)
    dense_child_hits = _merge_hits_by_chunk(dense_all)

    # 第二步：把 child 命中映射回 parent。
    # 这样后面的 fusion / rerank / generation 基于的就是更完整的 parent 上下文。
    sparse = _expand_hits_to_parent_chunks(runtime, sparse_child_hits)
    dense = _expand_hits_to_parent_chunks(runtime, dense_child_hits)

    t2 = time.perf_counter()
    fused = runtime.fusion.fuse(sparse, dense)
    RETRIEVAL_LATENCY.labels(stage="fusion").observe(time.perf_counter() - t2)
    if not fused:
        EMPTY_RETRIEVAL.inc()
    return {
        "sparse_hits": _dump(sparse),
        "dense_hits": _dump(dense),
        "fused_hits": _dump(fused),
    }
