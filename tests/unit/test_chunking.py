"""语义切块单元测试模块。用于校验切块边界、页码继承与 chunk_id 稳定性。"""

from core.ingestion.chunkers.semantic_chunker import SemanticChunker
from core.ingestion.metadata_extractors.basic import BasicMetadataExtractor
from core.models.document import CHILD_CHUNK_LEVEL, PARENT_CHUNK_LEVEL, Document


def test_semantic_chunker_splits_by_heading() -> None:
    doc = Document(
        doc_id="d1",
        source="s",
        title="T",
        content="## Intro\n\nPara one.\n\n## Details\n\nPara two is longer " + ("x" * 200),
    )
    ch = SemanticChunker(max_chars=120, overlap=10, min_chars=5)
    chunks = ch.chunk(doc)
    assert len(chunks) >= 2
    assert any("Intro" in (c.metadata.section or "") or "Intro" in c.content for c in chunks)


def test_semantic_chunker_builds_parent_child_structure() -> None:
    doc = Document(
        doc_id="doc-parent-child",
        source="s",
        title="Parent Child",
        content="## Section\n\n" + ("alpha beta gamma delta " * 80),
    )
    chunker = SemanticChunker(max_chars=80, overlap=10, min_chars=5, parent_max_chars=180)
    chunks = chunker.chunk(doc)

    parent_chunks = [c for c in chunks if c.metadata.chunk_level == PARENT_CHUNK_LEVEL]
    child_chunks = [c for c in chunks if c.metadata.chunk_level == CHILD_CHUNK_LEVEL]

    assert parent_chunks
    assert child_chunks
    parent_ids = {c.metadata.chunk_id for c in parent_chunks}
    assert all(c.metadata.parent_chunk_id in parent_ids for c in child_chunks)


def test_metadata_extractor_and_chunker_propagate_retrieval_hints() -> None:
    doc = Document(
        doc_id="schedule-doc",
        source="s",
        title="新疆能源（集团）有限责任公司 冲压二车间夜班值班安排表 V2.1",
        content=(
            "文件编号：XJNY-SC-2025-015\n"
            "生效日期：2025-04-08\n"
            "失效日期：2026-04-08\n"
            "发布部门：生产运营部\n"
            "审批人：李四\n"
            "项目名称：一号机组检修项目\n"
            "适用区域：新疆区域\n"
            "适用场站：准东二矿\n"
            "值班人：张三\n"
            "3号线夜班巡检安排。\n"
            "## 巡检步骤\n"
            "1. 检查输煤皮带运行状态。\n"
            "2. 检查设备编号：EQ-19A。\n"
        ),
        metadata={
            "project_ids": ["proj-a", "proj-b"],
            "allowed_departments": ["生产运营部", "设备管理部"],
        },
    )
    extractor = BasicMetadataExtractor()
    enriched = extractor.enrich_retrieval_metadata("冲压二车间夜班值班安排表.md", doc)
    chunks = SemanticChunker(max_chars=80, overlap=10, min_chars=5).chunk(enriched)
    child = next(
        c
        for c in chunks
        if c.metadata.chunk_level == CHILD_CHUNK_LEVEL and c.metadata.section == "巡检步骤"
    )
    assert child.metadata.extra.get("department") == "冲压二车间"
    assert child.metadata.extra.get("shift") == "夜班"
    assert child.metadata.extra.get("line") == "3号线"
    assert child.metadata.extra.get("person") == "张三"
    assert child.metadata.extra.get("owner_department") == "冲压二车间"
    assert child.metadata.extra.get("group_company") == "新疆能源（集团）有限责任公司"
    assert child.metadata.extra.get("doc_number") == "XJNY-SC-2025-015"
    assert child.metadata.extra.get("doc_type") == "markdown"
    assert child.metadata.extra.get("data_classification") == "internal"
    assert child.metadata.extra.get("authority_level") == "medium"
    assert child.metadata.extra.get("version") == "2.1"
    assert child.metadata.extra.get("version_status") == "active"
    assert child.metadata.extra.get("status") == "active"
    assert child.metadata.extra.get("effective_date") == "2025-04-08"
    assert child.metadata.extra.get("expiry_date") == "2026-04-08"
    assert child.metadata.extra.get("source_system") == "local_file"
    assert child.metadata.extra.get("issued_by") == "生产运营部"
    assert child.metadata.extra.get("approved_by") == "李四"
    assert child.metadata.extra.get("business_domain") == "equipment_maintenance"
    assert child.metadata.extra.get("process_stage") == "inspection"
    assert child.metadata.extra.get("equipment_id") == "EQ-19A"
    assert child.metadata.extra.get("project_name") == "一号机组检修项目"
    assert child.metadata.extra.get("applicable_region") == "新疆区域"
    assert child.metadata.extra.get("applicable_site") == "准东二矿"
    assert child.metadata.extra.get("section_type") in {"procedure", "general"}
    assert child.metadata.extra.get("section_level") == "2"
    assert child.metadata.extra.get("contains_steps") == "true"
    assert child.metadata.extra.get("contains_version_signal") == "true"
    assert isinstance(child.metadata.extra.get("topic_keywords"), list)
    assert child.metadata.extra.get("project_ids") == ["proj-a", "proj-b"]
    assert child.metadata.extra.get("allowed_departments") == ["生产运营部", "设备管理部"]


def test_pdf_chunker_uses_page_sections_for_unheaded_pdf() -> None:
    doc = Document(
        doc_id="pdf-doc",
        source="s",
        title="PDF Doc",
        content="<!-- page:1 -->\n第一页内容。\n\n<!-- page:2 -->\n第二页内容。",
        mime_type="application/pdf",
        metadata={"doc_type": "pdf"},
    )
    chunks = SemanticChunker(max_chars=1200, overlap=150, min_chars=2).chunk(doc)
    parent_chunks = [c for c in chunks if c.metadata.chunk_level == PARENT_CHUNK_LEVEL]
    assert {c.metadata.page for c in parent_chunks} >= {1, 2}
    assert {c.metadata.section for c in parent_chunks} >= {"第1页", "第2页"}


def test_csv_chunker_uses_tighter_chunk_profile() -> None:
    doc = Document(
        doc_id="csv-doc",
        source="s",
        title="CSV Doc",
        content="## Row 1\n\n" + ("字段: 数值说明 " * 80),
        metadata={"doc_type": "csv"},
        mime_type="text/csv",
    )
    chunks = SemanticChunker(max_chars=1200, overlap=150, min_chars=5).chunk(doc)
    child_chunks = [c for c in chunks if c.metadata.chunk_level == CHILD_CHUNK_LEVEL]
    assert child_chunks
    assert max(len(c.content) for c in child_chunks) <= 220
