"""检索链路单元测试模块。覆盖 BM25、向量召回、融合与相关数据结构。"""

from types import SimpleNamespace

import numpy as np
from scipy.sparse import coo_array

from core.retrieval.metadata_filters import build_milvus_filter_expression
from core.models.document import ChunkMetadata, TextChunk
from core.retrieval.metadata_filters import chunk_matches_filters
from core.retrieval.bgem3_backend import sparse_row_to_milvus_dict
from core.retrieval.hybrid_fusion import HybridFusion, reciprocal_rank_fusion
from core.retrieval.sparse_retriever import SparseRetriever
from core.retrieval.schemas import RetrievedChunk


def _chunk(cid: str, text: str) -> TextChunk:
    return TextChunk(
        content=text,
        metadata=ChunkMetadata(
            doc_id="d",
            chunk_id=cid,
            source="src",
            title="t",
        ),
    )


def test_bm25_search_orders_results() -> None:
    chunks = [
        _chunk("a", "password reset flow for users"),
        _chunk("b", "billing and invoices"),
    ]
    settings = SimpleNamespace(
        bm25_top_k=5,
        retrieval_embedding_backend="classic",
        embedding_model_name="BAAI/bge-m3",
        bgem3_device="cpu",
        bgem3_use_fp16=False,
    )
    r = SparseRetriever(settings)
    r.rebuild(chunks)
    hits = r.search("reset password", top_k=2)
    assert hits[0].chunk_id == "a"


def test_rrf_fusion_merges_lists() -> None:
    m = ChunkMetadata(doc_id="d", chunk_id="x", source="s", title="t")
    a = RetrievedChunk(chunk_id="x", score=1.0, content="c", metadata=m, trace={})
    b = RetrievedChunk(chunk_id="y", score=0.5, content="d", metadata=m.model_copy(update={"chunk_id": "y"}), trace={})
    fused = reciprocal_rank_fusion([[a], [b]], top_k=5)
    ids = [f.chunk_id for f in fused]
    assert "x" in ids and "y" in ids


def test_hybrid_fusion_switches_to_weighted_for_configured_query_scene() -> None:
    settings = SimpleNamespace(
        hybrid_top_k=5,
        fusion_strategy="rrf",
        fusion_sparse_weight=0.7,
        weighted_fusion_query_scenes="policy_lookup,error_code_lookup",
        weighted_fusion_scene_weights="policy_lookup:0.65,error_code_lookup:0.75",
    )
    fusion = HybridFusion(settings)
    m = ChunkMetadata(doc_id="d", chunk_id="x", source="s", title="t")
    sparse = [
        RetrievedChunk(
            chunk_id="x",
            score=10.0,
            content="policy text",
            metadata=m,
            trace={"retriever": "sparse"},
        )
    ]
    dense = [
        RetrievedChunk(
            chunk_id="y",
            score=0.9,
            content="dense text",
            metadata=m.model_copy(update={"chunk_id": "y"}),
            trace={"retriever": "dense"},
        )
    ]

    fused = fusion.fuse(sparse, dense, query_scene="policy_lookup")

    assert fused[0].trace["fusion"] == "weighted"
    assert fused[0].trace["fusion_strategy"] == "weighted"
    assert fused[0].trace["fusion_sparse_weight"] == 0.65
    assert fused[0].trace["query_scene"] == "policy_lookup"


def test_chunk_matches_filters_supports_list_metadata() -> None:
    metadata = ChunkMetadata(
        doc_id="d",
        chunk_id="c",
        source="s",
        title="t",
        extra={
            "project_ids": ["proj-a", "proj-b"],
            "allowed_roles": ["engineer", "manager"],
        },
    )
    assert chunk_matches_filters(metadata, {"project_ids": ["proj-a"]})
    assert chunk_matches_filters(metadata, {"allowed_roles": "engineer"})
    assert not chunk_matches_filters(metadata, {"project_ids": ["proj-x"]})


def test_chunk_matches_filters_supports_enterprise_scalar_metadata() -> None:
    metadata = ChunkMetadata(
        doc_id="d",
        chunk_id="c",
        source="s",
        title="t",
        extra={
            "business_domain": "equipment_maintenance",
            "process_stage": "inspection",
            "equipment_type": "输煤皮带",
            "section_type": "procedure",
        },
    )
    assert chunk_matches_filters(metadata, {"business_domain": "equipment_maintenance"})
    assert chunk_matches_filters(metadata, {"process_stage": "inspection"})
    assert chunk_matches_filters(metadata, {"equipment_type": "输煤皮带"})
    assert chunk_matches_filters(metadata, {"section_type": "procedure"})
    assert not chunk_matches_filters(metadata, {"business_domain": "finance"})


def test_chunk_matches_filters_treats_department_scope_as_or_group() -> None:
    metadata = ChunkMetadata(
        doc_id="d",
        chunk_id="c",
        source="s",
        title="t",
        extra={
            "owner_department": "安全环保部",
        },
    )
    assert chunk_matches_filters(
        metadata,
        {
            "department": "安全环保部",
            "owner_department": "安全环保部",
        },
    )


def test_build_milvus_filter_expression_uses_or_for_department_and_site_scope() -> None:
    expr = build_milvus_filter_expression(
        {
            "department": "安全环保部",
            "owner_department": "安全环保部",
            "plant": "准东二矿",
            "applicable_site": "准东二矿",
            "business_domain": "safety_production",
        }
    )

    assert 'searchable == true' in expr
    assert '(department == "安全环保部" or owner_department == "安全环保部")' in expr
    assert '(plant == "准东二矿" or applicable_site == "准东二矿")' in expr
    assert 'business_domain == "safety_production"' in expr


def test_sparse_retriever_supports_bgem3_sparse_backend(monkeypatch) -> None:
    class _FakeBGEM3:
        enabled = True

        def get_function(self):
            return object()

        def encode_documents(self, texts):
            _ = texts
            return {
                "sparse": np.asarray(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                )
            }

        def encode_queries(self, texts):
            _ = texts
            return {"sparse": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)}

    settings = SimpleNamespace(
        bm25_top_k=5,
        retrieval_embedding_backend="bgem3",
        embedding_model_name="BAAI/bge-m3",
        bgem3_device="cpu",
        bgem3_use_fp16=False,
    )
    retriever = SparseRetriever(settings)
    monkeypatch.setattr(retriever, "_bgem3", _FakeBGEM3())
    retriever.rebuild(
        [
            _chunk("a", "设备巡检流程"),
            _chunk("b", "财务报销流程"),
        ]
    )
    hits = retriever.search("设备巡检", top_k=2)
    assert hits[0].chunk_id == "a"
    assert hits[0].trace["retriever"] == "bgem3_sparse"


def test_sparse_row_to_milvus_dict_supports_coo_sparse_row() -> None:
    row = coo_array(([0.2, 0.5], ([0, 0], [3, 9])), shape=(1, 16))

    out = sparse_row_to_milvus_dict(row)

    assert out == {3: 0.2, 9: 0.5}


def test_sparse_retriever_prefers_milvus_sparse_when_enabled(monkeypatch) -> None:
    settings = SimpleNamespace(
        bm25_top_k=5,
        vector_backend="milvus",
        retrieval_embedding_backend="bgem3",
        embedding_model_name="./modes/bge-m3",
        bgem3_device="cpu",
        bgem3_use_fp16=False,
    )
    retriever = SparseRetriever(settings)

    class _FakeBGEM3:
        enabled = True

        def get_function(self):
            return object()

    class _FakeMilvus:
        def search_sparse(self, query, top_k=None, filters=None):
            _ = (query, top_k, filters)
            metadata = ChunkMetadata(doc_id="d", chunk_id="a", source="s", title="t")
            return [RetrievedChunk(chunk_id="a", score=1.0, content="设备巡检流程", metadata=metadata, trace={"retriever": "milvus_sparse"})]

    monkeypatch.setattr(retriever, "_bgem3", _FakeBGEM3())
    monkeypatch.setattr(retriever, "_milvus", _FakeMilvus())
    retriever.rebuild([_chunk("a", "设备巡检流程")])

    hits = retriever.search("设备巡检", top_k=2)

    assert hits[0].chunk_id == "a"
    assert hits[0].trace["retriever"] == "milvus_sparse"
