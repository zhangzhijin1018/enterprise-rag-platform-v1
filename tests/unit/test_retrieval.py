"""检索链路单元测试模块。覆盖 BM25、向量召回、融合与相关数据结构。"""

from core.models.document import ChunkMetadata, TextChunk
from core.retrieval.hybrid_fusion import reciprocal_rank_fusion
from core.retrieval.schemas import RetrievedChunk
from core.retrieval.sparse_retriever import SparseRetriever


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
    r = SparseRetriever()
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
