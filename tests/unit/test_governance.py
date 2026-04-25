"""企业知识治理排序与冲突检测单元测试。"""

from __future__ import annotations

import pytest

from core.models.document import ChunkMetadata
from core.orchestration.nodes.rerank_docs import rerank_docs_node
from core.retrieval.governance import apply_governance_ranking, detect_document_conflicts
from core.retrieval.schemas import RetrievedChunk


def _hit(
    chunk_id: str,
    *,
    score: float,
    title: str,
    version: str | None = None,
    effective_date: str | None = None,
    authority_level: str | None = None,
) -> RetrievedChunk:
    extra = {}
    if version is not None:
        extra["version"] = version
    if effective_date is not None:
        extra["effective_date"] = effective_date
    if authority_level is not None:
        extra["authority_level"] = authority_level
    metadata = ChunkMetadata(
        doc_id=f"doc-{chunk_id}",
        chunk_id=chunk_id,
        source=f"{chunk_id}.md",
        title=title,
        extra=extra,
    )
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        content=f"content-{chunk_id}",
        metadata=metadata,
        trace={},
    )


class _Settings:
    enable_governance_ranking = True
    authority_priority_boost = 0.08
    freshness_priority_boost = 0.06
    version_priority_boost = 0.04
    enable_conflict_detection = True
    conflict_detection_top_k = 5
    rerank_top_n = 8
    rerank_candidate_multiplier = 2
    rerank_candidate_max = 12


def test_apply_governance_ranking_prefers_newer_and_higher_authority_hits() -> None:
    settings = _Settings()
    older = _hit(
        "old",
        score=0.80,
        title="设备巡检管理制度",
        version="1.0",
        effective_date="2024-01-01",
        authority_level="low",
    )
    newer = _hit(
        "new",
        score=0.79,
        title="设备巡检管理制度",
        version="2.1",
        effective_date="2025-04-08",
        authority_level="high",
    )

    ranked = apply_governance_ranking([older, newer], settings)

    assert [item.chunk_id for item in ranked] == ["new", "old"]
    assert ranked[0].trace["semantic_score"] == pytest.approx(0.79)
    assert ranked[0].trace["governance_rank_score"] > ranked[1].trace["governance_rank_score"]


def test_detect_document_conflicts_reports_version_date_and_authority_gap() -> None:
    settings = _Settings()
    hits = [
        _hit(
            "preferred",
            score=0.88,
            title="设备巡检管理制度",
            version="2.1",
            effective_date="2025-04-08",
            authority_level="high",
        ),
        _hit(
            "legacy",
            score=0.87,
            title="设备巡检管理制度",
            version="1.0",
            effective_date="2024-01-01",
            authority_level="low",
        ),
    ]

    detected, summary = detect_document_conflicts(hits, settings)

    assert detected is True
    assert "版本不一致" in summary
    assert "生效日期不一致" in summary
    assert "权威级别不同" in summary
    assert "当前已优先采用" in summary


@pytest.mark.asyncio
async def test_rerank_docs_node_emits_conflict_fields_after_governance_sort() -> None:
    class _FakeReranker:
        def __init__(self) -> None:
            self.last_input_size = 0

        def rerank(self, question, hits, top_n=None):
            _ = (question, top_n)
            self.last_input_size = len(hits)
            return hits

    runtime = type("Runtime", (), {})()
    runtime.reranker = _FakeReranker()
    runtime.settings = _Settings()

    state = {
        "question": "最新的设备巡检制度是什么",
        "fused_hits": [
            _hit(
                "legacy",
                score=0.80,
                title="设备巡检管理制度",
                version="1.0",
                effective_date="2024-01-01",
                authority_level="low",
            ).model_dump(mode="json"),
            _hit(
                "preferred",
                score=0.79,
                title="设备巡检管理制度",
                version="2.1",
                effective_date="2025-04-08",
                authority_level="high",
            ).model_dump(mode="json"),
        ],
    }

    out = await rerank_docs_node(state, runtime)

    assert [item["chunk_id"] for item in out["reranked_hits"]] == ["preferred", "legacy"]
    assert out["conflict_detected"] is True
    assert "当前已优先采用" in out["conflict_summary"]
    assert out["answer_mode"] == "grounded_answer_with_conflict"
    assert runtime.reranker.last_input_size == 2
    assert out["reranked_hits"][0]["trace"]["rerank_candidate_limit"] == 2


@pytest.mark.asyncio
async def test_rerank_docs_node_limits_candidates_before_reranker() -> None:
    class _FakeReranker:
        def __init__(self) -> None:
            self.last_input_size = 0

        def rerank(self, question, hits, top_n=None):
            _ = (question, top_n)
            self.last_input_size = len(hits)
            return hits[:top_n]

    runtime = type("Runtime", (), {})()
    runtime.reranker = _FakeReranker()
    runtime.settings = _Settings()

    state = {
        "question": "设备巡检制度有哪些版本",
        "fused_hits": [
            _hit(
                f"chunk-{idx}",
                score=1.0 - idx * 0.01,
                title="设备巡检管理制度",
                version=f"1.{idx}",
                effective_date=f"2025-04-{idx + 1:02d}",
                authority_level="medium",
            ).model_dump(mode="json")
            for idx in range(20)
        ],
        "rerank_top_n": 4,
    }

    out = await rerank_docs_node(state, runtime)

    assert runtime.reranker.last_input_size == 8
    assert len(out["reranked_hits"]) == 4
    assert out["reranked_hits"][0]["trace"]["rerank_candidate_limit"] == 8
    assert out["reranked_hits"][0]["trace"]["rerank_input_size"] == 20
