"""Milvus 向量检索模块。

这个模块把第五轮增强里的 Milvus 能力正式接入现有项目。

设计目标不是推翻当前工程骨架，而是做一层“兼容当前主链路”的 dense backend：

1. 现有 API、LangGraph state、检索节点尽量不改
2. 把真正的向量召回切到 Milvus / Milvus Lite
3. 让 Milvus 逐步承担 parent 回扩与全量 chunk 扫描能力

换句话说，当前工程采用的是：

- Milvus：作为 dense retrieval 的正式执行引擎
- Milvus：同时逐步成为 chunk 原文与 metadata 的权威来源

这是一个偏工程化的折中：

- 在不推翻上层接口的前提下，把检索底座收敛到单一存储
- 同时保留当前项目透明、可读、易调试的优点
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core.config.settings import Settings, get_settings
from core.models.document import ChunkMetadata, TextChunk
from core.observability import get_logger
from core.retrieval.bgem3_backend import sparse_row_to_milvus_dict
from core.retrieval.dense_retriever import DenseRetriever
from core.retrieval.hybrid_fusion import HybridFusion
from core.retrieval.metadata_filters import (
    MILVUS_DIRECT_FILTER_FIELDS,
    build_milvus_filter_expression,
    chunk_matches_filters,
)
from core.retrieval.schemas import RetrievedChunk

logger = get_logger(__name__)

# 这一组长度配置对应 Milvus collection 里的 VARCHAR 字段。
# 为了方便后续维护，这里把“字段英文名 -> 长度”集中写在一起；
# 字段中文说明见下方 `_MILVUS_FIELD_LABELS`。
_MILVUS_VARCHAR_LENGTHS = {
    "chunk_id": 256,
    "doc_id": 256,
    "source": 2048,
    "title": 1024,
    "section": 1024,
    "chunk_level": 32,
    "parent_chunk_id": 256,
    "doc_number": 256,
    "department": 256,
    "owner_department": 256,
    "group_company": 256,
    "subsidiary": 256,
    "plant": 256,
    "shift": 64,
    "line": 128,
    "person": 128,
    "time": 128,
    "environment": 128,
    "version": 128,
    "version_status": 64,
    "doc_category": 128,
    "doc_type": 128,
    "status": 64,
    "data_classification": 64,
    "effective_date": 64,
    "expiry_date": 64,
    "authority_level": 64,
    "source_system": 128,
    "issued_by": 128,
    "approved_by": 128,
    "owner_role": 128,
    "business_domain": 128,
    "process_stage": 128,
    "applicable_region": 256,
    "applicable_site": 256,
    "equipment_type": 128,
    "equipment_id": 128,
    "system_name": 256,
    "project_name": 256,
    "project_phase": 128,
    "section_path": 1024,
    "section_level": 32,
    "section_type": 64,
    "contains_table": 16,
    "contains_steps": 16,
    "contains_contact": 16,
    "contains_version_signal": 16,
    "contains_risk_signal": 16,
}

# Milvus schema 字段中文说明。
# 主要用于：
# 1. 帮助快速理解 collection 里每个字段承载的业务语义
# 2. 让后续 schema 扩展时，先想清楚“这个字段到底在表达什么”
_MILVUS_FIELD_LABELS = {
    "chunk_id": "chunk 唯一标识",
    "doc_id": "文档唯一标识",
    "source": "来源文件或来源路径",
    "title": "文档标题",
    "page": "页码",
    "section": "章节标题",
    "chunk_level": "chunk 层级，parent 或 child",
    "parent_chunk_id": "父 chunk id",
    "doc_number": "文号/制度编号",
    "department": "部门",
    "owner_department": "归属部门",
    "group_company": "集团主体",
    "subsidiary": "子公司/二级单位",
    "plant": "厂站/矿区/装置区域",
    "shift": "班次",
    "line": "线别/产线/线路",
    "person": "人员",
    "time": "时间约束",
    "environment": "环境/运行环境",
    "version": "版本号",
    "version_status": "版本状态",
    "doc_category": "文档类别",
    "doc_type": "文档类型",
    "status": "文档状态",
    "data_classification": "数据分级",
    "effective_date": "生效日期",
    "expiry_date": "失效日期",
    "authority_level": "权威级别",
    "source_system": "来源系统",
    "issued_by": "发布方",
    "approved_by": "审批方",
    "owner_role": "归属角色",
    "business_domain": "业务域",
    "process_stage": "流程阶段",
    "applicable_region": "适用区域",
    "applicable_site": "适用场站",
    "equipment_type": "设备类型",
    "equipment_id": "设备编号",
    "system_name": "系统名称",
    "project_name": "项目名称",
    "project_phase": "项目阶段",
    "section_path": "章节路径",
    "section_level": "章节层级",
    "section_type": "章节类型",
    "contains_table": "是否包含表格",
    "contains_steps": "是否包含步骤说明",
    "contains_contact": "是否包含联系人信息",
    "contains_version_signal": "是否包含版本信号",
    "contains_risk_signal": "是否包含风险信号",
    "searchable": "是否参与直接召回",
    "content": "chunk 正文",
    "extra_json": "扩展 metadata JSON",
    "embedding": "向量 embedding",
    "sparse_embedding": "稀疏向量 embedding",
}

# 这些字段当前最常参与：
# - ACL / metadata filter 下推
# - 数据分级与治理判断
# - 企业实体约束
#
# 因此它们最适合补 Milvus 标量倒排索引（INVERTED），
# 让 hybrid search 在“先过滤、再召回”场景下更稳。
_MILVUS_SCALAR_INDEX_FIELDS = {
    "searchable",
    "doc_number",
    "department",
    "owner_department",
    "plant",
    "applicable_site",
    "business_domain",
    "process_stage",
    "equipment_type",
    "equipment_id",
    "system_name",
    "project_name",
    "data_classification",
    "authority_level",
    "source_system",
    "status",
    "version_status",
    "doc_type",
}


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

    def _required_collection_fields(self) -> set[str]:
        """返回当前版本 collection 至少应具备的字段。

        这里列出的字段就是当前 Milvus collection 的核心 schema。
        阅读时建议结合 `_MILVUS_FIELD_LABELS` 一起看，中文语义会更直观。
        """

        return {
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
            "embedding",
            *({"sparse_embedding"} if self._bgem3.enabled and self._bgem3.get_function() is not None else set()),
            *MILVUS_DIRECT_FILTER_FIELDS,
        }

    def _search_output_fields(self) -> list[str]:
        """统一维护检索/读取时需要从 Milvus 返回的字段。"""

        return [
            "chunk_id",
            "doc_id",
            "source",
            "title",
            "page",
            "section",
            "chunk_level",
            "parent_chunk_id",
            "searchable",
            *sorted(
                MILVUS_DIRECT_FILTER_FIELDS
                - {"doc_id", "source", "title", "section", "page", "chunk_level"}
            ),
            "content",
            "extra_json",
        ]

    def _available_output_fields(self) -> list[str]:
        """按当前 collection 实际 schema 过滤 output fields。

        这样旧 collection 缺字段时不会因为 `output_fields` 非法而直接打挂启动流程。
        """

        desired = self._search_output_fields()
        client = self._get_client()
        collection_name = self._collection_name()
        if not client.has_collection(collection_name=collection_name):
            return desired
        if not hasattr(client, "describe_collection"):
            return desired
        try:
            desc = client.describe_collection(collection_name=collection_name)
        except TypeError:
            desc = client.describe_collection(collection_name)
        except Exception:
            return desired
        existing = self._extract_field_names(desc)
        if not existing:
            return desired
        return [field for field in desired if field in existing]

    def _extract_field_names(self, desc: Any) -> set[str]:
        """兼容不同 MilvusClient 返回结构，尽量提取 schema field 名。"""

        fields: Any = None
        if isinstance(desc, dict):
            fields = desc.get("fields")
            if not fields and isinstance(desc.get("schema"), dict):
                fields = desc["schema"].get("fields")
        else:
            fields = getattr(desc, "fields", None)
            if fields is None:
                schema = getattr(desc, "schema", None)
                fields = getattr(schema, "fields", None) if schema is not None else None

        names: set[str] = set()
        if not isinstance(fields, list):
            return names
        for field in fields:
            if isinstance(field, dict):
                name = field.get("name") or field.get("field_name")
            else:
                name = getattr(field, "name", None) or getattr(field, "field_name", None)
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
        return names

    def _collection_requires_migration(self) -> bool:
        """判断现有 collection 是否缺少当前版本必需字段。"""

        client = self._get_client()
        collection_name = self._collection_name()
        if not client.has_collection(collection_name=collection_name):
            return False
        if not hasattr(client, "describe_collection"):
            return False

        try:
            desc = client.describe_collection(collection_name=collection_name)
        except TypeError:
            desc = client.describe_collection(collection_name)
        except Exception as exc:  # pragma: no cover - 依赖真实 Milvus client
            logger.warning(
                "failed to inspect milvus collection schema",
                extra={"collection_name": collection_name, "error": str(exc)},
            )
            return False

        field_names = self._extract_field_names(desc)
        if not field_names:
            return False
        missing = sorted(self._required_collection_fields() - field_names)
        if missing:
            logger.info(
                "milvus collection schema requires migration",
                extra={"collection_name": collection_name, "missing_fields": missing},
            )
            return True
        return False

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
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=_MILVUS_VARCHAR_LENGTHS["chunk_id"],
        )
        schema.add_field(
            field_name="doc_id",
            datatype=DataType.VARCHAR,
            max_length=_MILVUS_VARCHAR_LENGTHS["doc_id"],
        )
        schema.add_field(
            field_name="source",
            datatype=DataType.VARCHAR,
            max_length=_MILVUS_VARCHAR_LENGTHS["source"],
        )
        schema.add_field(
            field_name="title",
            datatype=DataType.VARCHAR,
            max_length=_MILVUS_VARCHAR_LENGTHS["title"],
        )
        schema.add_field(field_name="page", datatype=DataType.INT64)
        schema.add_field(
            field_name="section",
            datatype=DataType.VARCHAR,
            max_length=_MILVUS_VARCHAR_LENGTHS["section"],
        )
        schema.add_field(
            field_name="chunk_level",
            datatype=DataType.VARCHAR,
            max_length=_MILVUS_VARCHAR_LENGTHS["chunk_level"],
        )
        schema.add_field(
            field_name="parent_chunk_id",
            datatype=DataType.VARCHAR,
            max_length=_MILVUS_VARCHAR_LENGTHS["parent_chunk_id"],
        )
        schema.add_field(field_name="searchable", datatype=DataType.BOOL)
        for field_name in sorted(MILVUS_DIRECT_FILTER_FIELDS - {"doc_id", "source", "title", "section", "page", "chunk_level"}):
            schema.add_field(
                field_name=field_name,
                datatype=DataType.VARCHAR,
                max_length=_MILVUS_VARCHAR_LENGTHS[field_name],
            )
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="extra_json", datatype=DataType.JSON)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
        if self._bgem3.enabled and self._bgem3.get_function() is not None:
            schema.add_field(field_name="sparse_embedding", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self._settings.milvus_index_type,
            metric_type=self._settings.milvus_metric_type,
        )
        if self._bgem3.enabled and self._bgem3.get_function() is not None:
            index_params.add_index(
                field_name="sparse_embedding",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )
        for field_name in sorted(_MILVUS_SCALAR_INDEX_FIELDS):
            index_params.add_index(
                field_name=field_name,
                index_type="INVERTED",
            )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        self._collection_ready = True

    def _serialize_record(
        self,
        chunk: TextChunk,
        embedding: np.ndarray,
        *,
        sparse_embedding: dict[int, float] | None = None,
    ) -> dict[str, Any]:
        """把项目内的 `TextChunk` 转成 Milvus entity。

        注意几点：

        1. `page` 不能直接存 `None`，这里统一转成 `-1`
        2. `section` / `parent_chunk_id` 这类可空字符串统一降级成 `""`
        3. `extra` 保留为 JSON，避免把扩展字段全部打平成 schema
        4. 一级 Milvus 字段主要服务“可下推过滤 + 排障可读性”
        """

        extra = chunk.metadata.extra
        direct_fields = {
            key: str(extra.get(key) or "")
            for key in MILVUS_DIRECT_FILTER_FIELDS
            if key not in {"doc_id", "source", "title", "section", "page", "chunk_level"}
        }

        record = {
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
            **direct_fields,
        }
        if sparse_embedding:
            record["sparse_embedding"] = sparse_embedding
        return record

    def _build_metadata_from_entity(self, entity: dict[str, Any]) -> ChunkMetadata:
        """把 Milvus 返回的 entity 还原成项目内部 metadata。

        这里的核心思路是：
        - 检索时常用的字段保留为一级 Milvus 字段，便于服务端过滤
        - 项目内部继续统一还原为 `ChunkMetadata + extra`
        - 上层编排、引用、评测不需要直接感知 Milvus schema
        """

        page = entity.get("page")
        page_value = int(page) if isinstance(page, (int, float)) and int(page) >= 0 else None
        section = str(entity.get("section") or "").strip() or None
        parent_chunk_id = str(entity.get("parent_chunk_id") or "").strip()
        extra_json = entity.get("extra_json")
        extra = dict(extra_json) if isinstance(extra_json, dict) else {}
        for key in MILVUS_DIRECT_FILTER_FIELDS - {"doc_id", "source", "title", "section", "page", "chunk_level"}:
            value = str(entity.get(key) or "").strip()
            if value:
                extra[key] = value

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

    def _entity_to_text_chunk(self, entity: dict[str, Any]) -> TextChunk:
        """把 Milvus entity 还原成 `TextChunk`。"""

        metadata = self._build_metadata_from_entity(entity)
        return TextChunk(content=str(entity.get("content") or ""), metadata=metadata)

    def get_chunk_by_id(self, chunk_id: str) -> TextChunk | None:
        """按 chunk_id 从 Milvus 读取单个 chunk。

        这一步主要用于：
        - child 命中后的 parent 回扩
        - 统一从 Milvus 按 id 回查 chunk
        """

        if not chunk_id:
            return None
        client = self._get_client()
        collection_name = self._collection_name()
        if not client.has_collection(collection_name=collection_name):
            return None
        rows = client.get(
            collection_name=collection_name,
            ids=[chunk_id],
            output_fields=self._available_output_fields(),
        )
        if not rows:
            return None
        entity = dict(rows[0])
        if "chunk_id" not in entity:
            entity["chunk_id"] = chunk_id
        return self._entity_to_text_chunk(entity)

    def fetch_all_chunks(self, batch_size: int = 1000) -> list[TextChunk]:
        """从 Milvus 全量扫描 chunk。

        当前主要用于两类场景：
        - 服务启动 / reload 时重建 sparse 检索器内存视图
        - `/reindex` 时把 Milvus 作为唯一权威数据源重新编码 dense 向量
        """

        client = self._get_client()
        collection_name = self._collection_name()
        if not client.has_collection(collection_name=collection_name):
            return []

        out: list[TextChunk] = []
        iterator = client.query_iterator(
            collection_name=collection_name,
            batch_size=batch_size,
            limit=-1,
            filter='chunk_id != ""',
            output_fields=self._available_output_fields(),
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    out.append(self._entity_to_text_chunk(dict(row)))
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        return out

    def ensure_remote_index(self, chunks: Sequence[TextChunk], matrix: np.ndarray | None) -> None:
        """在远端 collection 缺失时，自动根据本地镜像补建 Milvus 索引。

        这个方法主要用于：

        - 进程启动后，希望在 collection 已存在时快速校验 schema
        - 如果 collection 缺字段，就触发一次全量重建
        """

        if matrix is None or not chunks:
            return
        client = self._get_client()
        collection_name = self._collection_name()
        if client.has_collection(collection_name=collection_name):
            self._collection_ready = True
            if self._collection_requires_migration():
                logger.info(
                "milvus collection schema is outdated, rebuilding from current snapshot",
                extra={"collection_name": collection_name},
            )
            self.sync_remote_index(chunks, matrix)
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

        sparse_vectors: list[dict[int, float]] | None = None
        if self._bgem3.enabled and self._bgem3.get_function() is not None:
            texts = [chunk.content for chunk in chunks]
            sparse_outputs = self._bgem3.encode_documents(texts)["sparse"]
            sparse_vectors = [sparse_row_to_milvus_dict(row) for row in sparse_outputs]

        batch: list[dict[str, Any]] = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, matrix, strict=True)):
            sparse_embedding = sparse_vectors[idx] if sparse_vectors is not None else None
            batch.append(self._serialize_record(chunk, embedding, sparse_embedding=sparse_embedding))
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
                "has_sparse_embedding": sparse_vectors is not None,
            },
        )
        self._collection_ready = True
        self.rebuild(chunks, matrix)

    def search_sparse(
        self,
        query: str,
        top_k: int | None = None,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """执行基于 Milvus `SPARSE_FLOAT_VECTOR` 的稀疏检索。"""

        if not self._bgem3.enabled or self._bgem3.get_function() is None:
            return []
        client = self._get_client()
        collection_name = self._collection_name()
        if not client.has_collection(collection_name=collection_name):
            return []

        query_sparse = self._bgem3.encode_queries([query])["sparse"]
        query_vector = sparse_row_to_milvus_dict(query_sparse[0])
        if not query_vector:
            return []

        k = top_k or self._settings.bm25_top_k
        raw = client.search(
            collection_name=collection_name,
            anns_field="sparse_embedding",
            data=[query_vector],
            limit=max(k, k * 3 if filters else k),
            filter=build_milvus_filter_expression(filters),
            output_fields=self._available_output_fields(),
            search_params={
                "metric_type": "IP",
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
            if not chunk_matches_filters(metadata, filters):
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=metadata.chunk_id,
                    score=float(item.get("distance", 0.0)),
                    content=str(entity.get("content") or ""),
                    metadata=metadata,
                    trace={"retriever": "milvus_sparse"},
                )
            )
        return out

    def search_hybrid(
        self,
        query: str,
        *,
        sparse_top_k: int,
        dense_top_k: int,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        query_scene: str | None = None,
    ) -> list[RetrievedChunk]:
        """执行 Milvus 原生 hybrid search。

        这里的目标不是替换项目上层所有融合逻辑，而是优先把：
        - BGEM3 sparse
        - dense vector
        收敛到 Milvus 原生 `hybrid_search` 上。

        上层仍然保留：
        - metadata boost
        - governance ranking
        - conflict detection
        """

        if not self._bgem3.enabled or self._bgem3.get_function() is None:
            return []

        client = self._get_client()
        collection_name = self._collection_name()
        if not client.has_collection(collection_name=collection_name):
            return []

        try:
            from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker
        except ModuleNotFoundError:
            return []

        dense_query_vector = self.embed_query(query).tolist()
        sparse_query = self._bgem3.encode_queries([query])["sparse"]
        sparse_query_vector = sparse_row_to_milvus_dict(sparse_query[0])
        if not sparse_query_vector:
            return []

        limit = top_k or self._settings.hybrid_top_k
        expr = build_milvus_filter_expression(filters)
        dense_req = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="embedding",
            param={
                "metric_type": self._settings.milvus_metric_type,
                "params": {},
            },
            limit=max(limit, dense_top_k),
            expr=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_embedding",
            param={
                "metric_type": "IP",
                "params": {},
            },
            limit=max(limit, sparse_top_k),
            expr=expr,
        )

        strategy, sparse_weight = HybridFusion(self._settings).resolve_policy(query_scene=query_scene)
        if strategy == "weighted":
            ranker = WeightedRanker(sparse_weight, 1.0 - sparse_weight)
        else:
            ranker = RRFRanker()

        raw = client.hybrid_search(
            collection_name=collection_name,
            reqs=[sparse_req, dense_req],
            ranker=ranker,
            limit=limit,
            output_fields=self._available_output_fields(),
        )

        rows = raw[0] if raw and isinstance(raw[0], list) else []
        out: list[RetrievedChunk] = []
        for item in rows:
            entity = item.get("entity") or {}
            if "chunk_id" not in entity and "id" in item:
                entity["chunk_id"] = item["id"]
            metadata = self._build_metadata_from_entity(entity)
            if not chunk_matches_filters(metadata, filters):
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=metadata.chunk_id,
                    score=float(item.get("distance", 0.0)),
                    content=str(entity.get("content") or ""),
                    metadata=metadata,
                    trace={
                        "retriever": "milvus_hybrid",
                        "fusion": strategy,
                        "fusion_strategy": strategy,
                        "fusion_sparse_weight": sparse_weight,
                        "query_scene": query_scene or "",
                    },
                )
            )
        return out

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """执行基于 Milvus 的向量召回。"""

        k = top_k or self._settings.dense_top_k
        client = self._get_client()
        collection_name = self._collection_name()
        if not client.has_collection(collection_name=collection_name):
            return []

        query_vector = self.embed_query(query).tolist()
        raw = client.search(
            collection_name=collection_name,
            anns_field="embedding",
            data=[query_vector],
            limit=max(k, k * 3 if filters else k),
            filter=build_milvus_filter_expression(filters),
            output_fields=self._available_output_fields(),
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
            if not chunk_matches_filters(metadata, filters):
                continue
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
