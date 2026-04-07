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

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        _ = top_k
        return [item.model_copy(deep=True) for item in self.mapping.get(query, [])]


class _FakeFusion:
    def fuse(
        self,
        sparse_hits: list[RetrievedChunk],
        dense_hits: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        return sparse_hits + dense_hits


class _FakeSettings:
    bm25_top_k = 5
    dense_top_k = 5


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


def _child_hit(chunk_id: str, parent_chunk_id: str, score: float, retriever: str) -> RetrievedChunk:
    base = _hit(chunk_id, score, retriever)
    return base.model_copy(
        update={
            "metadata": base.metadata.model_copy(
                update={"extra": {"parent_chunk_id": parent_chunk_id, "chunk_level": "child"}}
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
    runtime.sparse = _FakeRetriever(
        {
            "Milvus 和 Zilliz Cloud 有什么区别？": [_child_hit("a", "parent-a", 1.0, "bm25")],
            "Milvus 特点 适用场景": [
                _child_hit("a", "parent-a", 0.8, "bm25"),
                _child_hit("b", "parent-b", 0.7, "bm25"),
            ],
            "Milvus": [_child_hit("a", "parent-a", 0.9, "bm25")],
        }
    )
    runtime.dense = _FakeRetriever(
        {
            "Milvus 和 Zilliz Cloud 区别 对比 适用场景": [_child_hit("b", "parent-b", 0.9, "dense")],
            "Zilliz Cloud 特点 适用场景": [_child_hit("c", "parent-c", 0.85, "dense")],
        }
    )
    runtime.fusion = _FakeFusion()

    state = {
        "question": "Milvus 和 Zilliz Cloud 有什么区别？",
        "rewritten_query": "Milvus 和 Zilliz Cloud 区别 对比 适用场景",
        "multi_queries": ["Milvus 特点 适用场景", "Zilliz Cloud 特点 适用场景"],
        "keyword_queries": ["Milvus"],
        "hyde_query": "",
    }
    out = await retrieve_docs_node(state, runtime)
    sparse_ids = [item["chunk_id"] for item in out["sparse_hits"]]
    dense_ids = [item["chunk_id"] for item in out["dense_hits"]]
    assert "parent-a" in sparse_ids
    assert "parent-b" in sparse_ids
    assert "parent-b" in dense_ids
    assert "parent-c" in dense_ids
