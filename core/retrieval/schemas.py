"""检索结果数据模型模块。

`RetrievedChunk` 是检索、融合、重排和生成之间传递的核心结构。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.models.document import ChunkMetadata


class RetrievedChunk(BaseModel):
    """一次召回命中的标准表示。

    `trace` 字段用于记录该结果来自哪一路检索器、经过了什么融合策略，
    对调试 badcase 很有帮助。
    """

    chunk_id: str
    score: float
    content: str
    metadata: ChunkMetadata
    trace: dict[str, Any] = Field(default_factory=dict)
