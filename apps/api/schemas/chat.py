"""问答接口的数据模型模块。统一定义聊天请求、引用信息和返回结果的结构。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    top_k: int = Field(default=8, ge=1, le=64, description="Used as rerank top_n when provided")
    stream: bool = False


class CitationSchema(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    source: str
    page: int | None = None
    section: str | None = None


class RetrievedChunkSchema(BaseModel):
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    citations: list[CitationSchema]
    retrieved_chunks: list[RetrievedChunkSchema]
