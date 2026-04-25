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
        extra={
            "doc_type": "pdf",
            "owner_department": "信息化部",
            "data_classification": "internal",
            "version": "v2.1",
            "effective_date": "2025-04-08",
            "authority_level": "high",
            "source_system": "oa",
            "business_domain": "equipment_maintenance",
            "process_stage": "inspection",
            "section_path": "巡检管理/巡检步骤",
        },
    )
    hit = RetrievedChunk(
        chunk_id="c1",
        score=0.83,
        content="content",
        metadata=meta,
        trace={
            "matched_routes": ["original", "keyword_1"],
            "metadata_boost_reasons": ["intent:business_domain"],
            "semantic_score": 0.79,
            "governance_bonus": 0.12,
            "governance_rank_score": 0.91,
        },
    )
    c = chunk_to_citation(hit)
    assert isinstance(c, Citation)
    assert c.page == 3
    assert c.section == "Access"
    assert c.doc_type == "pdf"
    assert c.owner_department == "信息化部"
    assert c.data_classification == "internal"
    assert c.version == "v2.1"
    assert c.effective_date == "2025-04-08"
    assert c.authority_level == "high"
    assert c.source_system == "oa"
    assert c.business_domain == "equipment_maintenance"
    assert c.process_stage == "inspection"
    assert c.section_path == "巡检管理/巡检步骤"
    assert c.matched_routes == ["original", "keyword_1"]
    assert c.retrieval_score == 0.83
    assert c.semantic_score == 0.79
    assert c.governance_rank_score == 0.91
    assert c.selection_reason is not None
    assert "命中了 original / keyword_1 检索路线" in c.selection_reason


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
