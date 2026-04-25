"""检索结果数据模型模块。

`RetrievedChunk` 是 retrieval 层最核心的中间结果结构之一。

它会在这些阶段之间流转：
- 稀疏召回 / 稠密召回
- hybrid fusion
- governance / rerank
- generation 上下文组装

所以这里的字段设计重点不是“存更多东西”，而是保证：
- 上游召回结果能继续往下传
- 下游能知道这个片段从哪里来
- 调试时能复盘“它为什么会排到这里”
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.models.document import ChunkMetadata


class RetrievedChunk(BaseModel):
    """一次召回命中的标准表示。

    `trace` 字段用于记录该结果来自哪一路检索器、经过了什么融合策略，
    对调试 badcase 很有帮助。

    可以把它理解成：
    - `TextChunk` 更偏“索引中的静态知识片段”
    - `RetrievedChunk` 更偏“本次查询命中的动态结果对象”

    两者内容很像，但关注点不同：
    - `TextChunk` 关心可被索引
    - `RetrievedChunk` 关心可被排序、解释和继续传递
    """

    chunk_id: str = Field(description="chunk 唯一标识")
    score: float = Field(description="当前阶段分数；可能是召回分、融合分或治理排序后的分数")
    content: str = Field(description="chunk 正文")
    metadata: ChunkMetadata = Field(description="chunk 对应的来源与检索 metadata")
    trace: dict[str, Any] = Field(default_factory=dict, description="检索链路 trace，用于记录 route、boost、治理排序等调试信息")
