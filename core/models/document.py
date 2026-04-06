"""文档与 chunk 领域模型模块。

这些模型是入库、检索、引用三条链路之间的公共语言：
- `Document` 表示解析后的整篇文档；
- `TextChunk` 表示可索引的片段；
- `ChunkMetadata` 表示追踪来源、页码和 section 所需的信息。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """chunk 的检索与引用元数据。"""

    doc_id: str
    chunk_id: str
    source: str
    title: str = ""
    page: int | None = None
    section: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


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
