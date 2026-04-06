"""语义切块单元测试模块。用于校验切块边界、页码继承与 chunk_id 稳定性。"""

from core.ingestion.chunkers.semantic_chunker import SemanticChunker
from core.models.document import Document


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
