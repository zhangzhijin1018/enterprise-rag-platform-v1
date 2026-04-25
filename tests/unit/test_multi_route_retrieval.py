"""多路检索聚合单元测试。"""

from __future__ import annotations

import pytest

from core.models.document import ChunkMetadata
from core.orchestration.nodes.retrieve_docs import retrieve_docs_node
from core.retrieval.schemas import RetrievedChunk


class _FakeRetriever:
    """足够薄的假检索器。"""

    def __init__(self, mapping: dict[str, list[RetrievedChunk]]) -> None:
        self.mapping = mapping
        self.calls: list[tuple[str, int | None, dict | None]] = []

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, top_k, filters))
        return [item.model_copy(deep=True) for item in self.mapping.get(query, [])]


class _FakeFusion:
    def fuse(
        self,
        sparse_hits: list[RetrievedChunk],
        dense_hits: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        return sparse_hits + dense_hits


class _FakeHybridDense:
    def __init__(self, mapping: dict[str, list[RetrievedChunk]]) -> None:
        self.mapping = mapping
        self.calls: list[tuple[str, int, int, int, dict | None, str | None]] = []

    def search_hybrid(
        self,
        query: str,
        *,
        sparse_top_k: int,
        dense_top_k: int,
        top_k: int | None = None,
        filters: dict | None = None,
        query_scene: str | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, sparse_top_k, dense_top_k, int(top_k or 0), filters, query_scene))
        strategy = "weighted" if query_scene == "policy_lookup" else "rrf"
        out: list[RetrievedChunk] = []
        for item in self.mapping.get(query, []):
            trace = dict(item.trace)
            trace.setdefault("fusion", strategy)
            trace.setdefault("fusion_strategy", strategy)
            trace.setdefault("fusion_sparse_weight", 0.65 if strategy == "weighted" else 0.5)
            trace.setdefault("query_scene", query_scene or "")
            out.append(item.model_copy(update={"trace": trace}, deep=True))
        return out


class _FakeSettings:
    bm25_top_k = 5
    dense_top_k = 5
    hybrid_top_k = 8
    broad_sub_query_limit = 3
    balanced_sub_query_limit = 2
    keyword_route_limit = 2
    enable_acl = False
    enable_data_classification = True
    enable_model_routing = True
    default_data_classification = "internal"
    allow_external_llm_for_sensitive = False
    local_only_classifications = ["restricted"]
    acl_strict_mode = False
    metadata_match_boost = 0.08
    enterprise_entity_match_boost = 0.12
    structured_filter_boost = 0.05


class _FakeStore:
    def __init__(self, chunks: dict[str, RetrievedChunk]) -> None:
        self._chunks = chunks

    def get_chunk_by_id(self, chunk_id: str):
        item = self._chunks.get(chunk_id)
        if item is None:
            return None
        from core.models.document import TextChunk

        return TextChunk(content=item.content, metadata=item.metadata)


def _hit(chunk_id: str, score: float, retriever: str) -> RetrievedChunk:
    meta = ChunkMetadata(doc_id="doc", chunk_id=chunk_id, source="src", title="title")
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        content=f"content-{chunk_id}",
        metadata=meta,
        trace={"retriever": retriever},
    )


def _child_hit(
    chunk_id: str,
    parent_chunk_id: str,
    score: float,
    retriever: str,
    **extra,
) -> RetrievedChunk:
    base = _hit(chunk_id, score, retriever)
    metadata_extra = {"parent_chunk_id": parent_chunk_id, "chunk_level": "child"}
    metadata_extra.update(extra)
    return base.model_copy(
        update={
            "metadata": base.metadata.model_copy(
                update={"extra": metadata_extra}
            )
        }
    )


@pytest.mark.asyncio
async def test_retrieve_docs_node_merges_hits_from_multiple_query_routes() -> None:
    parent_a = _hit("parent-a", 0.0, "store")
    parent_b = _hit("parent-b", 0.0, "store")
    parent_c = _hit("parent-c", 0.0, "store")

    runtime = type("Runtime", (), {})()
    runtime.settings = _FakeSettings()
    runtime.store = _FakeStore(
        {
            "parent-a": parent_a,
            "parent-b": parent_b,
            "parent-c": parent_c,
        }
    )
    sparse = _FakeRetriever(
        {
            "Milvus 和 Zilliz Cloud 有什么区别？": [_child_hit("a", "parent-a", 1.0, "bm25")],
            "Milvus 和 Zilliz Cloud 的差异是什么": [_child_hit("c", "parent-c", 0.95, "bm25")],
            "Milvus 特点 适用场景": [
                _child_hit("a", "parent-a", 0.8, "bm25"),
                _child_hit("b", "parent-b", 0.7, "bm25"),
            ],
            "Milvus": [_child_hit("a", "parent-a", 0.9, "bm25")],
        }
    )
    dense = _FakeRetriever(
        {
            "Milvus 和 Zilliz Cloud 区别 对比 适用场景": [_child_hit("b", "parent-b", 0.9, "dense")],
            "Zilliz Cloud 特点 适用场景": [_child_hit("c", "parent-c", 0.85, "dense")],
        }
    )
    runtime.sparse = sparse
    runtime.dense = dense
    runtime.fusion = _FakeFusion()

    state = {
        "question": "Milvus 和 Zilliz Cloud 有什么区别？",
        "resolved_query": "Milvus 和 Zilliz Cloud 的差异是什么",
        "rewritten_query": "Milvus 和 Zilliz Cloud 区别 对比 适用场景",
        "multi_queries": ["Milvus 特点 适用场景", "Zilliz Cloud 特点 适用场景"],
        "keyword_queries": ["Milvus"],
        "hyde_query": "",
        "structured_filters": {"department": "平台架构部"},
    }
    out = await retrieve_docs_node(state, runtime)
    sparse_ids = [item["chunk_id"] for item in out["sparse_hits"]]
    dense_ids = [item["chunk_id"] for item in out["dense_hits"]]
    assert "parent-a" in sparse_ids
    assert "parent-b" in sparse_ids
    assert "parent-c" in sparse_ids
    assert "parent-b" in dense_ids
    assert "parent-c" in dense_ids
    assert any(query == "Milvus 和 Zilliz Cloud 的差异是什么" for query, _, _ in sparse.calls)
    assert all(filters == {"department": "平台架构部"} for _, _, filters in sparse.calls)
    assert all(filters == {"department": "平台架构部"} for _, _, filters in dense.calls)


@pytest.mark.asyncio
async def test_retrieve_docs_node_merges_access_filters_and_applies_acl() -> None:
    parent_a = _hit("parent-a", 0.0, "store")
    parent_b = _hit("parent-b", 0.0, "store")

    runtime = type("Runtime", (), {})()
    runtime.settings = _FakeSettings()
    runtime.settings.enable_acl = True
    runtime.store = _FakeStore({"parent-a": parent_a, "parent-b": parent_b})
    runtime.sparse = _FakeRetriever(
        {
            "设备巡检 SOP 在哪里": [
                _child_hit(
                    "a",
                    "parent-a",
                    1.0,
                    "bm25",
                    allowed_departments=["设备管理部"],
                    allowed_roles=["engineer"],
                    project_ids=["proj-a"],
                    data_classification="internal",
                ),
                _child_hit(
                    "b",
                    "parent-b",
                    0.9,
                    "bm25",
                    allowed_departments=["财务部"],
                    allowed_roles=["manager"],
                    project_ids=["proj-b"],
                    data_classification="restricted",
                ),
            ]
        }
    )
    runtime.dense = _FakeRetriever({})
    runtime.fusion = _FakeFusion()

    state = {
        "question": "设备巡检 SOP 在哪里",
        "structured_filters": {"department": "设备管理部"},
        "access_filters": {
            "department": "设备管理部",
            "role": "engineer",
            "project_ids": ["proj-a"],
            "clearance_level": "internal",
        },
        "user_context": {
            "department": "设备管理部",
            "role": "engineer",
            "project_ids": ["proj-a"],
            "clearance_level": "internal",
            "allow_external_llm": None,
        },
    }

    out = await retrieve_docs_node(state, runtime)

    assert [item["chunk_id"] for item in out["sparse_hits"]] == ["parent-a"]
    assert [item["chunk_id"] for item in out["fused_hits"]] == ["parent-a"]
    assert out["data_classification"] == "internal"
    assert out["model_route"] == "external_allowed"
    assert runtime.sparse.calls[0][2] == {
        "department": "设备管理部",
        "allowed_departments": "设备管理部",
        "allowed_roles": "engineer",
        "project_ids": ["proj-a"],
    }


@pytest.mark.asyncio
async def test_retrieve_docs_node_uses_precise_profile_and_metadata_boost() -> None:
    parent_a = _hit("parent-a", 0.0, "store")
    parent_b = _hit("parent-b", 0.0, "store")

    runtime = type("Runtime", (), {})()
    runtime.settings = _FakeSettings()
    runtime.store = _FakeStore({"parent-a": parent_a, "parent-b": parent_b})
    runtime.sparse = _FakeRetriever(
        {
            "设备巡检制度 Q/XJNY-2025-001 在哪里": [
                _child_hit(
                    "a",
                    "parent-a",
                    0.50,
                    "bm25",
                    business_domain="equipment_maintenance",
                    doc_number="Q/XJNY-2025-001",
                ),
                _child_hit(
                    "b",
                    "parent-b",
                    0.50,
                    "bm25",
                    business_domain="finance",
                ),
            ]
        }
    )
    runtime.dense = _FakeRetriever({})
    runtime.fusion = _FakeFusion()

    state = {
        "question": "设备巡检制度 Q/XJNY-2025-001 在哪里",
        "strategy_signals": {
            "top_k_profile": "precise",
            "preferred_retriever": "sparse",
            "metadata_intent": {
                "business_domain": "equipment_maintenance",
                "doc_number": "Q/XJNY-2025-001",
            },
        },
        "metadata_intent": {
            "business_domain": "equipment_maintenance",
            "doc_number": "Q/XJNY-2025-001",
        },
    }

    out = await retrieve_docs_node(state, runtime)

    assert [item["chunk_id"] for item in out["fused_hits"]] == ["parent-a", "parent-b"]
    assert out["fused_hits"][0]["trace"]["metadata_boost"] > out["fused_hits"][1]["trace"]["metadata_boost"]
    assert any("intent:business_domain" == reason for reason in out["fused_hits"][0]["trace"]["metadata_boost_reasons"])
    assert runtime.sparse.calls[0][1] == 4
    assert runtime.sparse.calls[0][2] == {
        "business_domain": "equipment_maintenance",
        "doc_number": "Q/XJNY-2025-001",
    }


@pytest.mark.asyncio
async def test_retrieve_docs_node_prunes_low_value_routes_for_precise_sparse_lookup() -> None:
    parent = _hit("parent-a", 0.0, "store")

    runtime = type("Runtime", (), {})()
    runtime.settings = _FakeSettings()
    runtime.store = _FakeStore({"parent-a": parent})
    runtime.sparse = _FakeRetriever({"设备巡检制度编号是什么": [_child_hit("a", "parent-a", 0.9, "bm25")]})
    runtime.dense = _FakeRetriever({})
    runtime.fusion = _FakeFusion()

    state = {
        "question": "设备巡检制度编号是什么",
        "resolved_query": "设备巡检制度编号是什么",
        "rewritten_query": "设备巡检制度 编号",
        "multi_queries": ["设备巡检制度 版本", "设备巡检制度 生效日期"],
        "keyword_queries": ["设备巡检制度", "制度编号", "设备巡检"],
        "hyde_query": "这是一份关于设备巡检制度编号的说明",
        "strategy_signals": {
            "top_k_profile": "precise",
            "preferred_retriever": "sparse",
        },
    }

    await retrieve_docs_node(state, runtime)

    sparse_queries = [query for query, _, _ in runtime.sparse.calls]
    dense_queries = [query for query, _, _ in runtime.dense.calls]
    assert "设备巡检制度编号是什么" in sparse_queries
    assert "设备巡检制度" in sparse_queries
    assert "制度编号" in sparse_queries
    assert "设备巡检制度 版本" not in sparse_queries
    assert "这是一份关于设备巡检制度编号的说明" not in dense_queries


@pytest.mark.asyncio
async def test_retrieve_docs_node_applies_enterprise_entity_boost_for_normalized_alias_intent() -> None:
    parent_a = _hit("parent-a", 0.0, "store")
    parent_b = _hit("parent-b", 0.0, "store")

    runtime = type("Runtime", (), {})()
    runtime.settings = _FakeSettings()
    runtime.store = _FakeStore({"parent-a": parent_a, "parent-b": parent_b})
    runtime.sparse = _FakeRetriever(
        {
            "安环部在二矿用安生平台看隐患排查记录怎么查": [
                _child_hit(
                    "a",
                    "parent-a",
                    0.50,
                    "bm25",
                    department="安全环保部",
                    owner_department="安全环保部",
                    plant="准东二矿",
                    system_name="安全生产管理平台",
                    business_domain="safety_production",
                ),
                _child_hit(
                    "b",
                    "parent-b",
                    0.54,
                    "bm25",
                    business_domain="safety_production",
                ),
            ]
        }
    )
    runtime.dense = _FakeRetriever({})
    runtime.fusion = _FakeFusion()

    state = {
        "question": "安环部在二矿用安生平台看隐患排查记录怎么查",
        "strategy_signals": {
            "top_k_profile": "balanced",
            "preferred_retriever": "hybrid",
            "metadata_intent": {
                "department": "安全环保部",
                "owner_department": "安全环保部",
                "plant": "准东二矿",
                "applicable_site": "准东二矿",
                "system_name": "安全生产管理平台",
                "business_domain": "safety_production",
            },
        },
        "metadata_intent": {
            "department": "安全环保部",
            "owner_department": "安全环保部",
            "plant": "准东二矿",
            "applicable_site": "准东二矿",
            "system_name": "安全生产管理平台",
            "business_domain": "safety_production",
        },
    }

    out = await retrieve_docs_node(state, runtime)

    assert [item["chunk_id"] for item in out["fused_hits"]] == ["parent-a", "parent-b"]
    trace = out["fused_hits"][0]["trace"]
    assert trace["enterprise_entity_boost"] == pytest.approx(0.48)
    assert trace["enterprise_entity_matches"] == [
        "department",
        "site",
        "system",
        "business_domain",
    ]
    assert "entity:department" in trace["metadata_boost_reasons"]
    assert "entity:site" in trace["metadata_boost_reasons"]
    assert "entity:system" in trace["metadata_boost_reasons"]


@pytest.mark.asyncio
async def test_retrieve_docs_node_prefers_native_milvus_hybrid_when_available() -> None:
    parent_a = _hit("parent-a", 0.0, "store")

    runtime = type("Runtime", (), {})()
    runtime.settings = _FakeSettings()
    runtime.store = _FakeStore({"parent-a": parent_a})
    runtime.sparse = _FakeRetriever({})
    runtime.dense = _FakeHybridDense(
        {
            "设备巡检制度 Q/XJNY-2025-001 在哪里": [
                _child_hit(
                    "a",
                    "parent-a",
                    0.91,
                    "milvus_hybrid",
                    business_domain="equipment_maintenance",
                    doc_number="Q/XJNY-2025-001",
                )
            ]
        }
    )
    runtime.fusion = _FakeFusion()

    state = {
        "question": "设备巡检制度 Q/XJNY-2025-001 在哪里",
        "query_scene": "policy_lookup",
        "strategy_signals": {
            "top_k_profile": "precise",
            "preferred_retriever": "hybrid",
            "metadata_intent": {
                "business_domain": "equipment_maintenance",
                "doc_number": "Q/XJNY-2025-001",
            },
        },
        "metadata_intent": {
            "business_domain": "equipment_maintenance",
            "doc_number": "Q/XJNY-2025-001",
        },
    }

    out = await retrieve_docs_node(state, runtime)

    assert [item["chunk_id"] for item in out["fused_hits"]] == ["parent-a"]
    assert out["fused_hits"][0]["trace"]["retriever"] == "milvus_hybrid"
    assert out["fused_hits"][0]["trace"]["fusion_strategy"] == "weighted"
    assert runtime.sparse.calls == []
    assert runtime.dense.calls[0][0] == "设备巡检制度 Q/XJNY-2025-001 在哪里"
    assert runtime.dense.calls[0][-1] == "policy_lookup"
