"""引用格式化单元测试模块。验证引用去重、结构转换和覆盖率计算逻辑。"""

from core.generation.citation_formatter import (
    Citation,
    chunk_to_citation,
    citation_coverage,
    format_citations_from_chunks,
)
from core.models.document import ChunkMetadata
from core.retrieval.schemas import RetrievedChunk


def test_chunk_to_citation_fields() -> None:
    meta = ChunkMetadata(
        doc_id="d1",
        chunk_id="c1",
        source="policy.pdf",
        title="Policy",
        page=3,
        section="Access",
    )
    c = chunk_to_citation(meta)
    assert isinstance(c, Citation)
    assert c.page == 3
    assert c.section == "Access"


def test_format_citations_from_chunks_dedup() -> None:
    m = ChunkMetadata(doc_id="d", chunk_id="c", source="s", title="t")
    r = RetrievedChunk(chunk_id="c", score=1.0, content="x", metadata=m, trace={})
    out = format_citations_from_chunks([r, r])
    assert len(out) == 1


def test_citation_coverage_ratio() -> None:
    c1 = Citation(doc_id="d", chunk_id="a", title="", source="")
    c2 = Citation(doc_id="d", chunk_id="b", title="", source="")
    cov = citation_coverage([c1], ["a", "b"])
    assert 0.0 < cov < 1.0
