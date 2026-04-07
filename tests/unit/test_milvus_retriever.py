"""Milvus 检索器单元测试。

这些测试不依赖真实 Milvus 服务，而是通过假客户端验证两件最关键的行为：

1. 全量同步时是否会重建 collection 并写入 entity
2. 搜索返回的 Milvus 结果，是否能正确还原成项目内部的 `RetrievedChunk`
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.config.settings import Settings
from core.models.document import ChunkMetadata, TextChunk
from core.retrieval.milvus_retriever import MilvusDenseRetriever


class _FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.indexes.append(kwargs)


class _FakeMilvusClient:
    def __init__(self) -> None:
        self.exists = False
        self.created = False
        self.dropped = False
        self.upserted: list[dict[str, Any]] = []
        self.search_rows: list[dict[str, Any]] = []

    def has_collection(self, *, collection_name: str) -> bool:
        _ = collection_name
        return self.exists

    def drop_collection(self, *, collection_name: str) -> None:
        _ = collection_name
        self.exists = False
        self.dropped = True

    def prepare_index_params(self) -> _FakeIndexParams:
        return _FakeIndexParams()

    def create_collection(self, **kwargs: Any) -> None:
        _ = kwargs
        self.exists = True
        self.created = True

    def upsert(self, *, collection_name: str, data: list[dict[str, Any]]) -> None:
        _ = collection_name
        self.upserted.extend(data)

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        _ = kwargs
        return [self.search_rows]


class _TestableMilvusDenseRetriever(MilvusDenseRetriever):
    def __init__(self, settings: Settings, client: _FakeMilvusClient) -> None:
        super().__init__(settings)
        self._fake_client = client

    def _create_client(self) -> Any:
        return self._fake_client

    def _ensure_collection(self, dim: int) -> None:
        _ = dim
        client = self._get_client()
        if not client.has_collection(collection_name=self._collection_name()):
            client.create_collection(collection_name=self._collection_name(), schema=None, index_params=None)
        self._collection_ready = True

    def embed_query(self, query: str) -> np.ndarray:
        _ = query
        return np.asarray([0.1, 0.2, 0.3], dtype=np.float32)


def _chunk(chunk_id: str, *, parent_chunk_id: str = "", level: str = "child") -> TextChunk:
    extra: dict[str, Any] = {"chunk_level": level}
    if parent_chunk_id:
        extra["parent_chunk_id"] = parent_chunk_id
    return TextChunk(
        content=f"content-{chunk_id}",
        metadata=ChunkMetadata(
            doc_id="doc-1",
            chunk_id=chunk_id,
            source="unit-test",
            title="Title",
            extra=extra,
        ),
    )


def test_milvus_sync_remote_index_recreates_collection_and_upserts_entities() -> None:
    client = _FakeMilvusClient()
    client.exists = True
    settings = Settings(
        VECTOR_BACKEND="milvus",
        MILVUS_URI="./tmp/milvus.db",
        MILVUS_COLLECTION_NAME="rag_chunks_test",
    )
    retriever = _TestableMilvusDenseRetriever(settings, client)

    chunks = [_chunk("child-1", parent_chunk_id="parent-1"), _chunk("parent-1", level="parent")]
    matrix = np.asarray([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype=np.float32)

    retriever.sync_remote_index(chunks, matrix)

    assert client.dropped is True
    assert client.created is True
    assert len(client.upserted) == 2
    assert client.upserted[0]["chunk_id"] == "child-1"
    assert client.upserted[0]["parent_chunk_id"] == "parent-1"
    assert client.upserted[1]["chunk_level"] == "parent"


def test_milvus_search_maps_hits_back_to_retrieved_chunks() -> None:
    client = _FakeMilvusClient()
    client.exists = True
    client.search_rows = [
        {
            "id": "child-1",
            "distance": 0.88,
            "entity": {
                "chunk_id": "child-1",
                "doc_id": "doc-1",
                "source": "unit-test",
                "title": "Title",
                "page": -1,
                "section": "",
                "chunk_level": "child",
                "parent_chunk_id": "parent-1",
                "searchable": True,
                "content": "child content",
                "extra_json": {"foo": "bar"},
            },
        }
    ]
    settings = Settings(
        VECTOR_BACKEND="milvus",
        MILVUS_URI="./tmp/milvus.db",
        MILVUS_COLLECTION_NAME="rag_chunks_test",
    )
    retriever = _TestableMilvusDenseRetriever(settings, client)
    retriever.rebuild([_chunk("child-1", parent_chunk_id="parent-1")], None)

    hits = retriever.search("what is milvus", top_k=3)

    assert len(hits) == 1
    assert hits[0].chunk_id == "child-1"
    assert hits[0].content == "child content"
    assert hits[0].trace["retriever"] == "milvus_dense"
    assert hits[0].metadata.parent_chunk_id == "parent-1"
    assert hits[0].metadata.extra["foo"] == "bar"
