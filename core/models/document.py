"""文档与 chunk 领域模型模块。

这些模型是入库、检索、引用三条链路之间的公共语言：
- `Document` 表示解析后的整篇文档；
- `TextChunk` 表示可索引的片段；
- `ChunkMetadata` 表示追踪来源、页码和 section 所需的信息。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

PARENT_CHUNK_LEVEL = "parent"
CHILD_CHUNK_LEVEL = "child"


class ChunkMetadata(BaseModel):
    """chunk 的检索与引用元数据。

    `extra` 字段保留了扩展空间，用来承载不适合立刻提升为一级字段、
    但又在特定链路里非常关键的附加信息。

    在第三轮增强里，`extra` 里会重点放这几类信息：

    - `chunk_level`：当前 chunk 是 `parent` 还是 `child`
    - `parent_chunk_id`：child chunk 对应的 parent chunk

    为什么不直接把这些字段一上来就提升为一级字段：

    1. 这样能保持当前 API schema 和磁盘结构的兼容性更好；
    2. 未来如果还要加 `tenant_id`、`acl_tag`、`table_id` 等扩展字段，
       仍然可以沿用同一套扩展槽位。
    """

    doc_id: str
    chunk_id: str
    source: str
    title: str = ""
    page: int | None = None
    section: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def chunk_level(self) -> str:
        """返回当前 chunk 的层级。

        兼容策略：
        - 老索引里没有 `chunk_level` 字段时，默认把它当作 `child`
        - 这样旧数据不会因为升级分层检索而全部失效
        """

        raw = self.extra.get("chunk_level")
        if raw == PARENT_CHUNK_LEVEL:
            return PARENT_CHUNK_LEVEL
        return CHILD_CHUNK_LEVEL

    @property
    def parent_chunk_id(self) -> str | None:
        """返回当前 chunk 对应的父 chunk id。"""

        raw = self.extra.get("parent_chunk_id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return None

    @property
    def is_parent(self) -> bool:
        """是否是 parent chunk。"""

        return self.chunk_level == PARENT_CHUNK_LEVEL

    @property
    def is_child(self) -> bool:
        """是否是 child chunk。"""

        return self.chunk_level == CHILD_CHUNK_LEVEL

    @property
    def searchable(self) -> bool:
        """当前 chunk 是否应该直接进入召回索引。

        当前策略：
        - parent chunk 主要用于给 rerank / generation 提供更完整上下文
        - child chunk 主要用于召回

        所以默认只有 child chunk 参与 BM25 / Dense 检索。
        """

        return self.is_child


class Document(BaseModel):
    """Normalized document after parsing."""

    doc_id: str
    source: str
    title: str = ""
    content: str
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextChunk(BaseModel):
    """可进入索引的最小文本单元。"""

    content: str
    metadata: ChunkMetadata

    @property
    def is_parent(self) -> bool:
        """代理到 metadata，便于上层代码少写一层访问。"""

        return self.metadata.is_parent

    @property
    def is_child(self) -> bool:
        """代理到 metadata，便于上层代码少写一层访问。"""

        return self.metadata.is_child

    @property
    def searchable(self) -> bool:
        """代理到 metadata，便于检索器统一过滤可搜索 chunk。"""

        return self.metadata.searchable
