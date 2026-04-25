"""检索节点模块。

该节点会同时执行：
1. BM25 稀疏召回
2. 向量稠密召回
3. 多路 query route 融合

当前支持的 route 至少包括：
- `original`：用户原始问题，作为基线路线
- `resolved`：多轮上下文补全后的完整问题
- `rewrite`：轻量检索改写后的主 query
- `sub_query`：复杂问题拆分后的子查询
- `keyword`：适合稀疏检索的关键词路线
- `hyde`：仅走 dense 的假设答案路线
"""

from __future__ import annotations

import inspect
import time
from typing import Any

from core.models.document import TextChunk
from core.observability import get_logger
from core.observability.metrics import EMPTY_RETRIEVAL, RETRIEVAL_LATENCY
from core.orchestration.state import RAGState
from core.retrieval.access_control import (
    build_retrieval_acl_filters,
    is_chunk_accessible,
    resolve_data_classification,
    resolve_model_route,
)
from core.retrieval.metadata_filters import chunk_matches_filters
from core.retrieval.schemas import RetrievedChunk
from core.security.risk_engine import (
    RuleBasedRiskEngine,
    build_risk_context,
    decision_to_state_update,
    safe_evaluate_risk,
)
from core.services.runtime import RAGRuntime

logger = get_logger(__name__)

_ENTERPRISE_ENTITY_GROUPS: dict[str, str] = {
    "department": "department",
    "owner_department": "department",
    "plant": "site",
    "applicable_site": "site",
    "system_name": "system",
    "business_domain": "business_domain",
    "process_stage": "process_stage",
    "equipment_type": "equipment",
    "equipment_id": "equipment",
    "project_name": "project",
}
# 企业 metadata 意图到“实体语义组”的映射。
# 用途：
# 1. fusion 后做 enterprise entity boost
# 2. trace 里保留更稳定的 explainability 语义


def _dump(hits: list[RetrievedChunk]) -> list[dict[str, Any]]:
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


def _clean_filters(filters: object) -> dict[str, Any]:
    """收敛结构化过滤条件，避免把空值透传到检索层。"""

    if not isinstance(filters, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in filters.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                out[key] = text
            continue
        if isinstance(value, list):
            vals = [str(item).strip() for item in value if str(item).strip()]
            if vals:
                out[key] = vals
            continue
        if value is not None:
            out[key] = value
    return out


def _annotate_hits(
    hits: list[RetrievedChunk],
    *,
    route_name: str,
    route_query: str,
    route_kind: str,
    structured_filters: dict[str, Any],
    access_filters: dict[str, Any],
) -> list[RetrievedChunk]:
    """给检索结果补充 query 路线 trace。"""

    out: list[RetrievedChunk] = []
    for hit in hits:
        trace = dict(hit.trace)
        trace["query_route"] = route_name
        trace["query_route_kind"] = route_kind
        trace["query_text"] = route_query[:256]
        trace["structured_filters"] = structured_filters
        trace["access_filters"] = access_filters
        out.append(hit.model_copy(update={"trace": trace}))
    return out


def _merge_filter_values(existing: Any, incoming: Any) -> Any:
    """合并两个 filter 值，兼容标量与列表。

    这里做的是“值级别去重”，不是字段级别覆盖。
    适合把 metadata_intent、structured_filters、ACL filter 合成一个最终过滤字典。
    """

    if existing is None:
        return incoming
    if incoming is None:
        return existing

    existing_values = existing if isinstance(existing, list) else [existing]
    incoming_values = incoming if isinstance(incoming, list) else [incoming]
    merged: list[Any] = []
    seen: set[str] = set()
    for item in [*existing_values, *incoming_values]:
        key = repr(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    if len(merged) == 1:
        return merged[0]
    return merged


def _merge_filters(*filters_list: dict[str, Any]) -> dict[str, Any]:
    """把多来源过滤条件收敛成单个过滤字典。"""

    merged: dict[str, Any] = {}
    for filters in filters_list:
        for key, value in filters.items():
            merged[key] = _merge_filter_values(merged.get(key), value)
    return merged


def _merge_query_filters(
    *,
    structured_filters: dict[str, Any],
    metadata_intent: dict[str, Any],
    access_filters: dict[str, Any],
) -> dict[str, Any]:
    """合并 query 侧过滤条件。

    规则：
    - `metadata_intent` 作为轻量检索意图，优先补默认过滤
    - `structured_filters` 视为显式用户约束，同名键覆盖意图值
    - `access_filters` 始终叠加，不能被 query 侧条件覆盖
    """

    merged_query_filters = dict(metadata_intent)
    merged_query_filters.update(structured_filters)
    return _merge_filters(merged_query_filters, access_filters)


def _resolve_route_top_ks(
    *,
    state: RAGState,
    runtime: RAGRuntime,
    default_sparse: int,
    default_dense: int,
) -> tuple[int, int]:
    """按 query scene 决定本轮检索使用的 top_k。

    核心目标：
    - `precise`：少召回、少浪费
    - `broad`：给复杂问题更宽的候选面
    - `preferred_retriever`：进一步调整 sparse / dense 的资源分配
    """

    signals = state.get("strategy_signals") or {}
    profile = str(signals.get("top_k_profile") or "balanced")
    preferred_retriever = str(signals.get("preferred_retriever") or "hybrid")
    sparse_top_k = max(1, int(default_sparse))
    dense_top_k = max(1, int(default_dense))

    if profile == "precise":
        sparse_top_k = max(4, min(sparse_top_k, max(4, sparse_top_k // 2)))
        dense_top_k = max(2, min(dense_top_k, max(2, dense_top_k // 2)))
    elif profile == "broad":
        sparse_top_k = min(runtime.settings.hybrid_top_k, max(sparse_top_k, int(sparse_top_k * 1.5)))
        dense_top_k = min(runtime.settings.hybrid_top_k, max(dense_top_k, int(dense_top_k * 1.5)))

    if preferred_retriever == "sparse":
        dense_top_k = max(2, min(dense_top_k, max(2, dense_top_k // 2)))
    elif preferred_retriever == "dense":
        sparse_top_k = max(2, min(sparse_top_k, max(2, sparse_top_k // 2)))

    return sparse_top_k, dense_top_k


def _route_top_k(base_top_k: int, *, route_kind: str, retriever_kind: str) -> int:
    """进一步按 query route 缩放 top_k，避免每一路都跑满。

    例如：
    - `keyword` 路线更适合短而准的 sparse 召回
    - `hyde` 路线更适合作为 dense 补充，不必跑太宽
    """

    top_k = max(1, base_top_k)
    if route_kind == "keyword" and retriever_kind == "sparse":
        return max(3, min(top_k, max(3, top_k // 2)))
    if route_kind == "hyde" and retriever_kind == "dense":
        return max(3, min(top_k, max(3, top_k // 2)))
    return top_k


def _prune_query_routes(
    query_routes: list[tuple[str, str, str, bool, bool]],
    *,
    state: RAGState,
    runtime: RAGRuntime,
) -> list[tuple[str, str, str, bool, bool]]:
    """按 query profile 与 retriever 偏好裁剪 route，减少低价值召回。

    这一步的作用不是“少功能”，而是“少浪费”：
    - 精确检索不必把所有子查询和 HyDE 都跑一遍
    - sparse 偏好场景不必保留太多 dense 侧低价值路线
    """

    signals = state.get("strategy_signals") or {}
    profile = str(signals.get("top_k_profile") or "balanced")
    preferred_retriever = str(signals.get("preferred_retriever") or "hybrid")

    kept: list[tuple[str, str, str, bool, bool]] = []
    sub_query_count = 0
    keyword_count = 0

    for route_name, route_query, route_kind, use_sparse, use_dense in query_routes:
        if route_kind == "hyde" and profile != "broad":
            continue
        if route_kind == "sub_query":
            sub_query_count += 1
            if profile == "precise":
                continue
            limit = int(
                getattr(
                    runtime.settings,
                    "broad_sub_query_limit" if profile == "broad" else "balanced_sub_query_limit",
                    2,
                )
            )
            if sub_query_count > max(1, limit):
                continue
        if route_kind == "keyword":
            keyword_count += 1
            if keyword_count > max(1, int(getattr(runtime.settings, "keyword_route_limit", 2))):
                continue

        sparse_flag = use_sparse
        dense_flag = use_dense
        if preferred_retriever == "sparse":
            if route_kind in {"sub_query", "keyword"}:
                dense_flag = False
            if profile == "precise" and route_kind in {"rewritten_query", "resolved_query"}:
                dense_flag = False
        elif preferred_retriever == "dense":
            if route_kind == "keyword":
                continue
            if route_kind == "sub_query":
                sparse_flag = False

        if not sparse_flag and not dense_flag:
            continue
        kept.append((route_name, route_query, route_kind, sparse_flag, dense_flag))

    return kept or [query_routes[0]]


def _filter_accessible_hits(
    hits: list[RetrievedChunk],
    *,
    user_context: dict[str, Any],
    runtime: RAGRuntime,
) -> list[RetrievedChunk]:
    """基于 chunk metadata 做 ACL / 数据分级兜底过滤。"""

    out: list[RetrievedChunk] = []
    for hit in hits:
        if not is_chunk_accessible(hit.metadata, user_context, runtime.settings):
            continue
        trace = dict(hit.trace)
        trace["acl_applied"] = True
        trace["data_classification"] = hit.metadata.extra.get(
            "data_classification",
            runtime.settings.default_data_classification,
        )
        out.append(hit.model_copy(update={"trace": trace}))
    return out


def _boost_hits_by_metadata(
    hits: list[RetrievedChunk],
    *,
    metadata_intent: dict[str, Any],
    structured_filters: dict[str, Any],
    runtime: RAGRuntime,
) -> list[RetrievedChunk]:
    """在融合后按 metadata 命中情况做轻量预加权。"""

    if not hits:
        return []

    boosted: list[tuple[float, int, RetrievedChunk]] = []
    for index, hit in enumerate(hits):
        base_score = float(hit.score)
        bonus = 0.0
        reasons: list[str] = []
        matched_entity_groups: list[str] = []
        matched_entity_keys: list[str] = []
        for key, value in metadata_intent.items():
            if chunk_matches_filters(hit.metadata, {key: value}):
                bonus += float(runtime.settings.metadata_match_boost)
                reasons.append(f"intent:{key}")
                entity_group = _ENTERPRISE_ENTITY_GROUPS.get(key)
                if entity_group and entity_group not in matched_entity_groups:
                    matched_entity_groups.append(entity_group)
                    matched_entity_keys.append(key)
        for key, value in structured_filters.items():
            if chunk_matches_filters(hit.metadata, {key: value}):
                bonus += float(runtime.settings.structured_filter_boost)
                reasons.append(f"filter:{key}")
        entity_bonus = len(matched_entity_groups) * float(
            getattr(runtime.settings, "enterprise_entity_match_boost", 0.0)
        )
        if entity_bonus > 0:
            bonus += entity_bonus
            reasons.extend(f"entity:{group}" for group in matched_entity_groups)
        final_score = base_score + bonus
        trace = dict(hit.trace)
        trace["base_retrieval_score"] = round(base_score, 6)
        trace["metadata_boost"] = round(bonus, 6)
        trace["enterprise_entity_boost"] = round(entity_bonus, 6)
        trace["enterprise_entity_matches"] = matched_entity_groups
        trace["enterprise_entity_matched_keys"] = matched_entity_keys
        trace["metadata_boost_reasons"] = reasons
        trace["boosted_retrieval_score"] = round(final_score, 6)
        boosted.append(
            (
                final_score,
                -index,
                hit.model_copy(update={"score": final_score, "trace": trace}),
            )
        )
    boosted.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in boosted]


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
    """根据 child 命中结果找到它对应的 parent chunk。"""

    chunk_loader = getattr(runtime.dense, "get_chunk_by_id", None)
    if not callable(chunk_loader):
        store = getattr(runtime, "store", None)
        chunk_loader = getattr(store, "get_chunk_by_id", None)
    if not callable(chunk_loader):
        return None
    if hit.metadata.is_parent:
        return chunk_loader(hit.chunk_id)
    parent_chunk_id = hit.metadata.parent_chunk_id
    if not parent_chunk_id:
        return None
    return chunk_loader(parent_chunk_id)


def _merge_parent_child_metadata(hit: RetrievedChunk, parent_chunk: TextChunk) -> TextChunk:
    """回扩 parent 时保留 child 上已有的企业 metadata。"""

    merged_extra = dict(parent_chunk.metadata.extra)
    for key, value in hit.metadata.extra.items():
        if key in {"chunk_level", "parent_chunk_id"}:
            continue
        if key not in merged_extra or merged_extra[key] in (None, "", [], {}):
            merged_extra[key] = value
    return TextChunk(
        content=parent_chunk.content,
        metadata=parent_chunk.metadata.model_copy(update={"extra": merged_extra}),
    )


def _expand_hits_to_parent_chunks(
    runtime: RAGRuntime,
    hits: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """把 child 命中结果回扩成 parent 命中结果。"""

    merged: dict[str, RetrievedChunk] = {}
    child_ids_map: dict[str, list[str]] = {}
    routes_map: dict[str, list[str]] = {}
    route_kind_map: dict[str, list[str]] = {}
    for hit in hits:
        parent_chunk = _resolve_parent_chunk(runtime, hit)
        if parent_chunk is None:
            parent_chunk = TextChunk(content=hit.content, metadata=hit.metadata)
        else:
            parent_chunk = _merge_parent_child_metadata(hit, parent_chunk)
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

    logger.info("retrieval started", extra={"event": "retrieval_started"})
    base_sk = state.get("top_k_sparse") or runtime.settings.bm25_top_k
    base_dk = state.get("top_k_dense") or runtime.settings.dense_top_k
    sk, dk = _resolve_route_top_ks(
        state=state,
        runtime=runtime,
        default_sparse=base_sk,
        default_dense=base_dk,
    )
    question = state.get("question") or ""
    resolved = state.get("resolved_query") or ""
    rewritten = state.get("rewritten_query") or ""
    multi_queries = state.get("multi_queries") or []
    keyword_queries = state.get("keyword_queries") or []
    hyde_query = state.get("hyde_query") or ""
    strategy_signals = dict(state.get("strategy_signals") or {})
    structured_filters = _clean_filters(state.get("structured_filters"))
    metadata_intent = _clean_filters(
        state.get("metadata_intent") or strategy_signals.get("metadata_intent")
    )
    user_context = dict(state.get("user_context") or {})
    access_filters = _clean_filters(state.get("access_filters"))
    retrieval_acl_filters = {}
    if getattr(runtime.settings, "enable_acl", False):
        retrieval_acl_filters = _clean_filters(build_retrieval_acl_filters(access_filters))
    retrieval_filters = _merge_query_filters(
        structured_filters=structured_filters,
        metadata_intent=metadata_intent,
        access_filters=retrieval_acl_filters,
    )

    query_routes: list[tuple[str, str, str, bool, bool]] = [
        ("original", question, "direct_query", True, True)
    ]
    if resolved.strip() and resolved.strip() != question.strip():
        query_routes.append(("resolved", resolved, "resolved_query", True, True))
    if rewritten.strip() and rewritten.strip() not in {question.strip(), resolved.strip()}:
        query_routes.append(("rewrite", rewritten, "rewritten_query", True, True))
    for idx, sub_query in enumerate(_dedupe_queries(multi_queries), start=1):
        query_routes.append((f"sub_query_{idx}", sub_query, "sub_query", True, True))
    for idx, keyword_query in enumerate(_dedupe_queries(keyword_queries), start=1):
        query_routes.append((f"keyword_{idx}", keyword_query, "keyword", True, False))
    if hyde_query.strip():
        query_routes.append(("hyde", hyde_query, "hyde", False, True))
    query_routes = _prune_query_routes(query_routes, state=state, runtime=runtime)

    sparse_all: list[RetrievedChunk] = []
    dense_all: list[RetrievedChunk] = []
    native_hybrid_all: list[RetrievedChunk] = []
    sparse_elapsed = 0.0
    dense_elapsed = 0.0
    hybrid_elapsed = 0.0
    for route_name, route_query, route_kind, use_sparse, use_dense in query_routes:
        if use_sparse and use_dense:
            hybrid_search = getattr(runtime.dense, "search_hybrid", None)
            if callable(hybrid_search):
                t_hybrid = time.perf_counter()
                hybrid_hits = hybrid_search(
                    route_query,
                    sparse_top_k=_route_top_k(sk, route_kind=route_kind, retriever_kind="sparse"),
                    dense_top_k=_route_top_k(dk, route_kind=route_kind, retriever_kind="dense"),
                    top_k=runtime.settings.hybrid_top_k,
                    filters=retrieval_filters,
                    query_scene=str(state.get("query_scene") or ""),
                )
                hybrid_elapsed += time.perf_counter() - t_hybrid
                native_hybrid_all.extend(
                    _annotate_hits(
                        hybrid_hits,
                        route_name=route_name,
                        route_query=route_query,
                        route_kind=route_kind,
                        structured_filters=structured_filters,
                        access_filters=access_filters,
                    )
                )
                continue
        if use_sparse:
            t_sparse = time.perf_counter()
            sparse_hits = runtime.sparse.search(
                route_query,
                top_k=_route_top_k(sk, route_kind=route_kind, retriever_kind="sparse"),
                filters=retrieval_filters,
            )
            sparse_elapsed += time.perf_counter() - t_sparse
            sparse_all.extend(
                _annotate_hits(
                    sparse_hits,
                    route_name=route_name,
                    route_query=route_query,
                    route_kind=route_kind,
                    structured_filters=structured_filters,
                    access_filters=access_filters,
                )
            )
        if use_dense:
            t_dense = time.perf_counter()
            dense_hits = runtime.dense.search(
                route_query,
                top_k=_route_top_k(dk, route_kind=route_kind, retriever_kind="dense"),
                filters=retrieval_filters,
            )
            dense_elapsed += time.perf_counter() - t_dense
            dense_all.extend(
                _annotate_hits(
                    dense_hits,
                    route_name=route_name,
                    route_query=route_query,
                    route_kind=route_kind,
                    structured_filters=structured_filters,
                    access_filters=access_filters,
                )
            )
    RETRIEVAL_LATENCY.labels(stage="sparse").observe(sparse_elapsed)
    RETRIEVAL_LATENCY.labels(stage="dense").observe(dense_elapsed)
    RETRIEVAL_LATENCY.labels(stage="hybrid").observe(hybrid_elapsed)

    sparse_raw_count = len(sparse_all)
    dense_raw_count = len(dense_all)
    native_hybrid_raw_count = len(native_hybrid_all)
    sparse_child_hits = _filter_accessible_hits(
        _merge_hits_by_chunk(sparse_all),
        user_context=user_context,
        runtime=runtime,
    )
    dense_child_hits = _filter_accessible_hits(
        _merge_hits_by_chunk(dense_all),
        user_context=user_context,
        runtime=runtime,
    )

    sparse = _expand_hits_to_parent_chunks(runtime, sparse_child_hits)
    dense = _expand_hits_to_parent_chunks(runtime, dense_child_hits)
    native_hybrid = _expand_hits_to_parent_chunks(
        runtime,
        _filter_accessible_hits(
            _merge_hits_by_chunk(native_hybrid_all),
            user_context=user_context,
            runtime=runtime,
        ),
    )

    t2 = time.perf_counter()
    fusion_kwargs: dict[str, Any] = {}
    try:
        if "query_scene" in inspect.signature(runtime.fusion.fuse).parameters:
            fusion_kwargs["query_scene"] = str(state.get("query_scene") or "")
    except (TypeError, ValueError):
        # 测试桩对象或某些动态绑定对象可能拿不到完整签名，此时直接走旧调用方式。
        pass
    fused = runtime.fusion.fuse(sparse, dense, **fusion_kwargs)
    if native_hybrid:
        fused = _merge_hits_by_chunk(native_hybrid + fused)
    fused = _boost_hits_by_metadata(
        fused,
        metadata_intent=metadata_intent,
        structured_filters=structured_filters,
        runtime=runtime,
    )
    RETRIEVAL_LATENCY.labels(stage="fusion").observe(time.perf_counter() - t2)
    data_classification = resolve_data_classification(
        [item.metadata for item in fused],
        default=runtime.settings.default_data_classification,
    )
    model_route = resolve_model_route(
        settings=runtime.settings,
        data_classification=data_classification,
        allow_external_llm=user_context.get("allow_external_llm"),
    )
    risk_context = build_risk_context(
        stage="retrieval",
        question=question,
        audit_id=str(state.get("audit_id") or ""),
        user_context=user_context,
        state={
            **state,
            "reranked_hits": _dump(fused),
            "data_classification": data_classification,
            "model_route": model_route,
        },
    )
    risk_engine = getattr(runtime, "risk_engine", RuleBasedRiskEngine(runtime.settings))
    risk_decision = safe_evaluate_risk(risk_engine, risk_context, runtime.settings)
    risk_state = decision_to_state_update(risk_decision)
    if risk_decision.force_local_only:
        model_route = "local_only"
    response: RAGState = {
        "sparse_hits": _dump(sparse),
        "dense_hits": _dump(dense),
        "fused_hits": _dump(fused),
        "data_classification": data_classification,
        "model_route": model_route,
        "answer_mode": "grounded_answer",
        "metadata_intent": metadata_intent,
        **risk_state,
    }
    if not risk_decision.allow:
        response.update(
            {
                "fused_hits": [],
                "refusal": True,
                "refusal_reason": risk_decision.reason or "risk_engine_denied",
                "answer": "当前请求命中了企业风控策略，已拒绝继续处理。",
                "reasoning_summary": "检索阶段风控引擎判定为高风险请求。",
                "answer_mode": "refusal",
            }
        )
        logger.info(
            "retrieval denied by risk engine",
            extra={
                "event": "retrieval_completed",
                "retrieved_chunks": 0,
                "data_classification": data_classification,
                "model_route": model_route,
                "refusal": True,
            },
        )
        return response
    if not fused:
        EMPTY_RETRIEVAL.inc()
        if (sparse_raw_count or dense_raw_count or native_hybrid_raw_count) and not (
            sparse_child_hits or dense_child_hits or native_hybrid
        ):
            response.update(
                {
                    "refusal": True,
                    "refusal_reason": "access_denied",
                    "answer": "当前用户无权访问相关知识，无法提供答案。",
                    "reasoning_summary": "检索命中后因 ACL / 数据分级过滤被全部拦截。",
                    "answer_mode": "refusal",
                }
            )
    logger.info(
        "retrieval completed",
        extra={
            "event": "retrieval_completed",
            "retrieved_chunks": len(fused),
            "data_classification": data_classification,
            "model_route": model_route,
            "refusal": bool(response.get("refusal")),
        },
    )
    return response
