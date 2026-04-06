"""引用格式化模块。

负责在“检索元数据”和“API / 前端可消费的引用结构”之间做转换。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.models.document import ChunkMetadata
from core.retrieval.schemas import RetrievedChunk


class Citation(BaseModel):
    """对外暴露的引用结构。"""

    doc_id: str
    chunk_id: str
    title: str
    source: str
    page: int | None = None
    section: str | None = None


def chunk_to_citation(meta: ChunkMetadata) -> Citation:
    """把 chunk metadata 转成引用对象。"""

    return Citation(
        doc_id=meta.doc_id,
        chunk_id=meta.chunk_id,
        title=meta.title,
        source=meta.source,
        page=meta.page,
        section=meta.section,
    )


def format_citations_from_chunks(chunks: list[RetrievedChunk]) -> list[Citation]:
    """把检索结果列表去重后转成引用列表。"""

    seen: set[str] = set()
    out: list[Citation] = []
    for c in chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        out.append(chunk_to_citation(c.metadata))
    return out


def citation_coverage(citations: list[Citation], retrieved_ids: list[str]) -> float:
    """计算“召回到的 chunk 里有多少真正被引用到了”。"""

    if not retrieved_ids:
        return 0.0
    cited = {c.chunk_id for c in citations}
    return len(cited & set(retrieved_ids)) / max(1, len(set(retrieved_ids)))
