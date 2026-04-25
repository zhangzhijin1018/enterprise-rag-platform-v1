"""生成前上下文压缩单元测试。"""

from __future__ import annotations

from core.models.document import ChunkMetadata
from core.generation.context_format import select_contexts_for_prompt
from core.retrieval.schemas import RetrievedChunk


def _hit(
    chunk_id: str,
    *,
    doc_id: str,
    score: float,
    content: str,
    section_path: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        content=content,
        metadata=ChunkMetadata(
            doc_id=doc_id,
            chunk_id=chunk_id,
            source=f"{doc_id}.md",
            title=doc_id,
            extra={"section_path": section_path},
        ),
        trace={},
    )


def test_select_contexts_for_prompt_limits_docs_chunks_and_duplicate_sections() -> None:
    contexts = [
        _hit("a1", doc_id="doc-a", score=0.95, content="A" * 120, section_path="总则/范围"),
        _hit("a2", doc_id="doc-a", score=0.94, content="B" * 120, section_path="总则/范围"),
        _hit("a3", doc_id="doc-a", score=0.93, content="C" * 120, section_path="流程/步骤"),
        _hit("b1", doc_id="doc-b", score=0.92, content="D" * 120, section_path="职责/岗位"),
        _hit("c1", doc_id="doc-c", score=0.91, content="E" * 120, section_path="附录/表格"),
    ]

    packed = select_contexts_for_prompt(
        contexts,
        max_docs=2,
        max_chunks_per_doc=2,
        max_chars=900,
    )

    packed_ids = [item.chunk_id for item in packed]
    assert packed_ids == ["a1", "a3", "b1"]


def test_select_contexts_for_prompt_falls_back_to_first_when_budget_too_small() -> None:
    contexts = [
        _hit("a1", doc_id="doc-a", score=0.95, content="A" * 500, section_path="总则/范围"),
        _hit("b1", doc_id="doc-b", score=0.92, content="B" * 500, section_path="职责/岗位"),
    ]

    packed = select_contexts_for_prompt(
        contexts,
        max_docs=1,
        max_chunks_per_doc=1,
        max_chars=100,
    )

    assert [item.chunk_id for item in packed] == ["a1"]
