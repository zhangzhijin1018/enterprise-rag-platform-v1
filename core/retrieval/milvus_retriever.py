"""Milvus 向量检索模块。

这个模块把第五轮增强里的 Milvus 能力正式接入现有项目。

设计目标不是推翻当前工程骨架，而是做一层“兼容当前主链路”的 dense backend：

1. 现有 API、LangGraph state、检索节点尽量不改
2. 继续保留本地 `chunks.jsonl`，方便 BM25、parent 回扩和调试
3. 把真正的向量召回切到 Milvus / Milvus Lite

换句话说，当前工程采用的是：

- 本地 `IndexStore`：作为可读镜像、BM25 语料来源、parent chunk 回查来源
- Milvus：作为 dense retrieval 的正式执行引擎

这是一个偏工程化的折中：

- 保留当前项目透明、可读、易调试的优点
- 同时把“向量检索”这一层升级到专门数据库
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core.config.settings import Settings, get_settings
from core.models.document import ChunkMetadata, TextChunk
from core.observability import get_logger
from core.retrieval.dense_retriever import DenseRetriever
from core.retrieval.schemas import RetrievedChunk

logger = get_logger(__name__)


class MilvusDenseRetriever(DenseRetriever):
    """基于 Milvus 的稠密检索器。

    为什么继承 `DenseRetriever`：

    - 可以复用 embedding 模型加载逻辑
    - 对上层保持相同方法名：`rebuild / search / embed_query / embed_documents`
    - 让 runtime 装配只需要切换实例类型，而不必改动编排层调用方式
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self._client: Any | None = None
        self._collection_ready = False

    @property
    def backend_name(self) -> str:
        """返回当前 dense backend 名称，便于日志和调试。"""

        return "milvus"

    def _create_client(self) -> Any:
        """创建 MilvusClient。

        这里刻意延迟导入 `pymilvus`：

        - 没有使用 Milvus backend 时，不应该因为依赖缺失影响整个项目启动
        - 单元测试也可以通过 monkeypatch 这里，避免真实连接外部服务
        """

        try:
            from pymilvus import MilvusClient
        except ModuleNotFoundError as exc:  # pragma: no cover - 依赖缺失属于环境问题
            raise RuntimeError(
                "Milvus backend requires `pymilvus`. Install it before enabling VECTOR_BACKEND=milvus."
            ) from exc

        uri = self._settings.milvus_uri
        # 使用 Milvus Lite 本地文件时，先确保父目录存在，避免首次启动直接失败。
        if not uri.startswith("http://") and not uri.startswith("https://"):
            Path(uri).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = {"uri": uri}
        if self._settings.milvus_token:
            kwargs["token"] = self._settings.milvus_token
        if self._settings.milvus_db_name:
            kwargs["db_name"] = self._settings.milvus_db_name
        return MilvusClient(**kwargs)

    def _get_client(self) -> Any:
        """懒加载 MilvusClient。"""

        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _collection_name(self) -> str:
        """统一读取 collection 名称，避免硬编码。"""

        return self._settings.milvus_collection_name

    def _ensure_collection(self, dim: int) -> None:
        """确保 Milvus collection 已按当前 embedding 维度建好。

        当前策略非常明确：

        - 把“全量重建远端索引”当成一个显式动作
        - `sync_remote_index()` 会直接 drop + recreate collection
        - 这里主要负责首次建表

        这么做的好处是实现简单、状态清晰、排障成本低。
        对于当前项目这种教学型/工程骨架型代码，比复杂的增量 schema 演进更稳。
        """

        client = self._get_client()
        collection_name = self._collection_name()
        if client.has_collection(collection_name=collection_name):
            self._collection_ready = True
            return

        try:
            from pymilvus import DataType, MilvusClient
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "Milvus backend requires `pymilvus`. Install it before enabling VECTOR_BACKEND=milvus."
            ) from exc

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="page", datatype=DataType.INT64)
        schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="chunk_level", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="parent_chunk_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="searchable", datatype=DataType.BOOL)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="extra_json", datatype=DataType.JSON)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self._settings.milvus_index_type,
            metric_type=self._settings.milvus_metric_type,
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        self._collection_ready = True

    def _serialize_record(self, chunk: TextChunk, embedding: np.ndarray) -> dict[str, Any]:
        """把项目内的 `TextChunk` 转成 Milvus entity。

        注意几点：

        1. `page` 不能直接存 `None`，这里统一转成 `-1`
        2. `section` / `parent_chunk_id` 这类可空字符串统一降级成 `""`
        3. `extra` 保留为 JSON，避免把扩展字段全部打平成 schema
        """

        return {
            "chunk_id": chunk.metadata.chunk_id,
            "doc_id": chunk.metadata.doc_id,
            "source": chunk.metadata.source,
            "title": chunk.metadata.title or "",
            "page": chunk.metadata.page if chunk.metadata.page is not None else -1,
            "section": chunk.metadata.section or "",
            "chunk_level": chunk.metadata.chunk_level,
            "parent_chunk_id": chunk.metadata.parent_chunk_id or "",
            "searchable": chunk.searchable,
            "content": chunk.content,
            "extra_json": chunk.metadata.extra,
            "embedding": embedding.tolist(),
        }

    def _build_metadata_from_entity(self, entity: dict[str, Any]) -> ChunkMetadata:
        """把 Milvus 返回的 entity 还原成项目内部 metadata。"""

        page = entity.get("page")
        page_value = int(page) if isinstance(page, (int, float)) and int(page) >= 0 else None
        section = str(entity.get("section") or "").strip() or None
        parent_chunk_id = str(entity.get("parent_chunk_id") or "").strip()
        extra_json = entity.get("extra_json")
        extra = dict(extra_json) if isinstance(extra_json, dict) else {}

        # 这里显式把分层检索需要的两个关键信息补回 `extra`，
        # 这样上层代码无需感知 Milvus 的字段组织方式。
        extra["chunk_level"] = str(entity.get("chunk_level") or "child")
        if parent_chunk_id:
            extra["parent_chunk_id"] = parent_chunk_id

        return ChunkMetadata(
            doc_id=str(entity.get("doc_id") or ""),
            chunk_id=str(entity.get("chunk_id") or entity.get("id") or ""),
            source=str(entity.get("source") or ""),
            title=str(entity.get("title") or ""),
            page=page_value,
            section=section,
            extra=extra,
        )

    def rebuild(self, chunks: Sequence[TextChunk], matrix: np.ndarray | None) -> None:
        """刷新本地可搜索 chunk 视图，但不主动重写远端集合。

        这里和文件型 dense backend 的一个关键差别是：

        - 文件型 backend 的“矩阵”就在本地内存里，`rebuild()` 就能直接完成重建
        - Milvus backend 的真正索引在外部数据库里，`rebuild()` 只刷新本地状态
        - 真正的数据同步动作交给 `sync_remote_index()` 或 `ensure_remote_index()`
        """

        self._chunks = [c for c in chunks if c.searchable]
        self._matrix = None
        if not self._chunks:
            self._collection_ready = False

    def ensure_remote_index(self, chunks: Sequence[TextChunk], matrix: np.ndarray | None) -> None:
        """在远端 collection 缺失时，自动根据本地镜像补建 Milvus 索引。

        这个方法主要用于：

        - 进程重启后，本地已有 `chunks.jsonl + embeddings.npy`
        - 但 Milvus Lite 文件或远端 collection 还不存在
        - 希望系统启动时能自动把现有快照恢复到 Milvus
        """

        if matrix is None or not chunks:
            return
        client = self._get_client()
        collection_name = self._collection_name()
        if client.has_collection(collection_name=collection_name):
            self._collection_ready = True
            return
        self.sync_remote_index(chunks, matrix)

    def sync_remote_index(self, chunks: Sequence[TextChunk], matrix: np.ndarray | None) -> None:
        """把当前全量 chunk + 向量矩阵同步到 Milvus。

        当前采用“全量重建 collection”的方式，而不是复杂增量同步。

        原因：

        1. 当前项目的入库流程本来就会形成全量快照
        2. chunk_id 已经稳定，重建后的结果可预测、可验证
        3. 对教学型/骨架型项目而言，全量重建比双写增量同步更容易维护
        """

        if matrix is None:
            raise ValueError("Milvus sync requires embeddings matrix")
        if matrix.shape[0] != len(chunks):
            raise ValueError("embeddings must align with chunks before syncing to Milvus")
        if matrix.ndim != 2 or matrix.shape[1] <= 0:
            raise ValueError("Milvus sync requires a 2D embeddings matrix with positive dimension")

        client = self._get_client()
        collection_name = self._collection_name()
        if client.has_collection(collection_name=collection_name):
            client.drop_collection(collection_name=collection_name)
            self._collection_ready = False
        self._ensure_collection(dim=int(matrix.shape[1]))

        batch: list[dict[str, Any]] = []
        for chunk, embedding in zip(chunks, matrix, strict=True):
            batch.append(self._serialize_record(chunk, embedding))
            if len(batch) >= 128:
                client.upsert(collection_name=collection_name, data=batch)
                batch = []
        if batch:
            client.upsert(collection_name=collection_name, data=batch)

        logger.info(
            "milvus index synced",
            extra={
                "collection_name": collection_name,
                "num_chunks": len(chunks),
                "embedding_dim": int(matrix.shape[1]),
            },
        )
        self._collection_ready = True
        self.rebuild(chunks, matrix)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """执行基于 Milvus 的向量召回。"""

        k = top_k or self._settings.dense_top_k
        if not self._chunks:
            return []
        client = self._get_client()
        collection_name = self._collection_name()
        if not client.has_collection(collection_name=collection_name):
            return []

        query_vector = self.embed_query(query).tolist()
        raw = client.search(
            collection_name=collection_name,
            anns_field="embedding",
            data=[query_vector],
            limit=k,
            filter="searchable == true",
            output_fields=[
                "chunk_id",
                "doc_id",
                "source",
                "title",
                "page",
                "section",
                "chunk_level",
                "parent_chunk_id",
                "searchable",
                "content",
                "extra_json",
            ],
            search_params={
                "metric_type": self._settings.milvus_metric_type,
                "params": {},
            },
        )

        rows = raw[0] if raw and isinstance(raw[0], list) else []
        out: list[RetrievedChunk] = []
        for item in rows:
            entity = item.get("entity") or {}
            if "chunk_id" not in entity and "id" in item:
                entity["chunk_id"] = item["id"]
            metadata = self._build_metadata_from_entity(entity)
            out.append(
                RetrievedChunk(
                    chunk_id=metadata.chunk_id,
                    score=float(item.get("distance", 0.0)),
                    content=str(entity.get("content") or ""),
                    metadata=metadata,
                    trace={"retriever": "milvus_dense"},
                )
            )
        return out
