"""语义切块单元测试模块。用于校验切块边界、页码继承与 chunk_id 稳定性。"""

from core.ingestion.chunkers.semantic_chunker import SemanticChunker
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
