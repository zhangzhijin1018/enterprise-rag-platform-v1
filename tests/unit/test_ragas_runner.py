"""评测治理信号单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.evaluation.ragas_runner import (
    _answer_one,
    _build_eval_signal_summary,
    _build_query_understanding_report,
    _load_jsonl,
    _render_explainability_report,
    _state_eval_metadata,
)
from core.config.settings import Settings


def test_state_eval_metadata_extracts_governance_fields() -> None:
    state = {
        "refusal": True,
        "refusal_reason": "access_denied",
        "answer_mode": "refusal",
        "data_classification": "restricted",
        "model_route": "local_only",
        "conflict_detected": True,
        "conflict_summary": "版本不一致",
        "audit_id": "audit-123",
        "reranked_hits": [
            {
                "trace": {
                    "matched_routes": ["original", "keyword_1"],
                    "metadata_boost": 0.08,
                    "enterprise_entity_boost": 0.12,
                    "enterprise_entity_matches": ["department"],
                    "governance_bonus": 0.12,
                }
            }
        ],
        "citations": [
            {"chunk_id": "c1", "selection_reason": "命中了 original 检索路线"}
        ],
    }

    metadata = _state_eval_metadata(state)

    assert metadata == {
        "refusal": True,
        "refusal_reason": "access_denied",
        "answer_mode": "refusal",
        "data_classification": "restricted",
        "model_route": "local_only",
        "analysis_confidence": None,
        "analysis_source": None,
        "analysis_reason": None,
        "query_scene": None,
        "preferred_retriever": None,
        "top_k_profile": None,
        "conflict_detected": True,
        "conflict_summary": "版本不一致",
        "audit_id": "audit-123",
        "matched_routes": ["original", "keyword_1"],
        "metadata_boosted": True,
        "enterprise_entity_boosted": True,
        "enterprise_entity_matches": ["department"],
        "governance_boosted": True,
        "explainable_citation_count": 1,
    }


def test_build_eval_signal_summary_counts_refusal_conflict_and_routes() -> None:
    dataset_rows = [
        {
            "scenario": "policy_conflict",
            "tags": ["policy", "conflict"],
            "expected_refusal": False,
            "expected_conflict": False,
        },
        {
            "scenario": "permission_denied",
            "tags": ["acl", "security"],
            "expected_refusal": True,
            "expected_conflict": True,
        },
    ]
    rows = [
        {
            "refusal": False,
            "data_classification": "internal",
            "model_route": "external_allowed",
            "analysis_confidence": 0.91,
            "analysis_source": "heuristic",
            "analysis_reason": "命中文号",
            "query_scene": "policy_lookup",
            "preferred_retriever": "sparse",
            "top_k_profile": "precise",
            "conflict_detected": False,
            "matched_routes": ["original", "keyword_1"],
            "metadata_boosted": True,
            "enterprise_entity_boosted": True,
            "enterprise_entity_matches": ["department", "site"],
            "governance_boosted": False,
            "explainable_citation_count": 1,
        },
        {
            "refusal": True,
            "data_classification": "restricted",
            "model_route": "local_only",
            "analysis_confidence": 0.44,
            "analysis_source": "llm_enhanced_guardrail",
            "analysis_reason": "问题指代较强",
            "query_scene": "project_trace",
            "preferred_retriever": "hybrid",
            "top_k_profile": "balanced",
            "conflict_detected": True,
            "matched_routes": ["original"],
            "metadata_boosted": False,
            "enterprise_entity_boosted": False,
            "enterprise_entity_matches": [],
            "governance_boosted": True,
            "explainable_citation_count": 2,
        },
    ]

    summary = _build_eval_signal_summary(dataset_rows, rows, total=2)

    assert summary["sample_count"] == pytest.approx(2.0)
    assert summary["refusal_rate"] == pytest.approx(0.5)
    assert summary["conflict_detected_rate"] == pytest.approx(0.5)
    assert summary["metadata_boost_hit_rate"] == pytest.approx(0.5)
    assert summary["enterprise_entity_boost_hit_rate"] == pytest.approx(0.5)
    assert summary["governance_boost_hit_rate"] == pytest.approx(0.5)
    assert summary["avg_explainable_citations"] == pytest.approx(1.5)
    assert summary["avg_analysis_confidence"] == pytest.approx(0.675)
    assert summary["classification:internal"] == pytest.approx(0.5)
    assert summary["classification:restricted"] == pytest.approx(0.5)
    assert summary["model_route:external_allowed"] == pytest.approx(0.5)
    assert summary["model_route:local_only"] == pytest.approx(0.5)
    assert summary["analysis_source:heuristic"] == pytest.approx(0.5)
    assert summary["analysis_source:llm_enhanced_guardrail"] == pytest.approx(0.5)
    assert summary["matched_route:original"] == pytest.approx(1.0)
    assert summary["matched_route:keyword_1"] == pytest.approx(0.5)
    assert summary["entity_match:department"] == pytest.approx(0.5)
    assert summary["entity_match:site"] == pytest.approx(0.5)
    assert summary["scenario:policy_conflict"] == pytest.approx(0.5)
    assert summary["scenario:permission_denied"] == pytest.approx(0.5)
    assert summary["tag:policy"] == pytest.approx(0.5)
    assert summary["tag:conflict"] == pytest.approx(0.5)
    assert summary["tag:acl"] == pytest.approx(0.5)
    assert summary["expected_refusal_match_rate"] == pytest.approx(1.0)
    assert summary["expected_conflict_match_rate"] == pytest.approx(1.0)


def test_build_query_understanding_report_aggregates_scenarios_and_recommendations() -> None:
    rows = [
        {
            "scenario": "policy_conflict",
            "query_scene": "policy_lookup",
            "analysis_source": "heuristic",
            "analysis_confidence": 0.91,
        },
        {
            "scenario": "meeting_trace",
            "query_scene": "meeting_trace",
            "analysis_source": "llm_enhanced",
            "analysis_confidence": 0.55,
        },
        {
            "scenario": "meeting_trace",
            "query_scene": "meeting_trace",
            "analysis_source": "llm_enhanced_guardrail",
            "analysis_confidence": 0.42,
        },
    ]

    report = _build_query_understanding_report(rows)

    assert report["top_scenes"][0]["name"] == "meeting_trace"
    assert report["top_guardrail_scenarios"][0]["name"] == "meeting_trace"
    assert report["top_llm_enhanced_scenarios"][0]["name"] == "meeting_trace"
    assert report["recommendations"]


@pytest.mark.asyncio
async def test_answer_one_returns_answer_contexts_and_eval_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_run_rag_async(runtime, question):  # noqa: ANN001
        _ = runtime
        return {
            "answer": f"answer for {question}",
            "reranked_hits": [
                {"content": "ctx-a"},
                {"content": "ctx-b"},
            ],
            "refusal": False,
            "refusal_reason": "",
            "answer_mode": "grounded_answer_with_conflict",
            "data_classification": "internal",
            "model_route": "external_allowed",
            "analysis_confidence": 0.82,
            "analysis_source": "llm_enhanced",
            "analysis_reason": "问题包含版本冲突信号",
            "query_scene": "policy_lookup",
            "preferred_retriever": "hybrid",
            "top_k_profile": "broad",
            "conflict_detected": True,
            "conflict_summary": "版本不一致",
            "audit_id": "audit-456",
            "reranked_hits": [
                {
                    "content": "ctx-a",
                    "trace": {
                        "matched_routes": ["original", "rewrite"],
                        "metadata_boost": 0.08,
                        "enterprise_entity_boost": 0.12,
                        "enterprise_entity_matches": ["department"],
                        "governance_bonus": 0.12,
                    },
                },
                {"content": "ctx-b", "trace": {}},
            ],
            "citations": [
                {"chunk_id": "ctx-a", "selection_reason": "命中了 original / rewrite 检索路线"}
            ],
        }

    monkeypatch.setattr("core.evaluation.ragas_runner.run_rag_async", _fake_run_rag_async)

    answer, contexts, metadata = await _answer_one(runtime=object(), question="最新制度是什么")

    assert answer == "answer for 最新制度是什么"
    assert contexts == ["ctx-a", "ctx-b"]
    assert metadata["conflict_detected"] is True
    assert metadata["conflict_summary"] == "版本不一致"
    assert metadata["model_route"] == "external_allowed"
    assert metadata["analysis_source"] == "llm_enhanced"
    assert metadata["analysis_confidence"] == pytest.approx(0.82)
    assert metadata["query_scene"] == "policy_lookup"
    assert metadata["matched_routes"] == ["original", "rewrite"]
    assert metadata["metadata_boosted"] is True
    assert metadata["enterprise_entity_boosted"] is True
    assert metadata["enterprise_entity_matches"] == ["department"]
    assert metadata["governance_boosted"] is True
    assert metadata["explainable_citation_count"] == 1


def test_settings_default_eval_dataset_points_to_enterprise_eval() -> None:
    settings = Settings.model_construct(eval_dataset_path="./core/evaluation/datasets/enterprise_eval.jsonl")
    assert settings.eval_dataset_path.endswith("enterprise_eval.jsonl")


def test_enterprise_eval_dataset_is_well_formed() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "core/evaluation/datasets/enterprise_eval.jsonl"
    )
    rows = _load_jsonl(path)

    assert len(rows) >= 10
    assert any(row.get("scenario") == "policy_conflict" for row in rows)
    assert any(row.get("tags") for row in rows)
    assert all("question" in row for row in rows)


def test_render_explainability_report_includes_summary_and_badcases() -> None:
    payload = {
        "summary": {
            "sample_count": 2.0,
            "refusal_rate": 0.5,
            "conflict_detected_rate": 0.5,
            "metadata_boost_hit_rate": 0.5,
            "enterprise_entity_boost_hit_rate": 0.5,
            "governance_boost_hit_rate": 0.5,
            "avg_explainable_citations": 1.5,
            "avg_analysis_confidence": 0.7,
            "analysis_source:heuristic": 0.5,
            "matched_route:original": 1.0,
            "entity_match:department": 0.5,
        },
        "query_understanding_report": {
            "top_scenes": [{"name": "policy_lookup", "count": 1}],
            "top_guardrail_scenarios": [],
            "top_llm_enhanced_scenarios": [{"name": "policy_conflict", "count": 1}],
            "recommendations": ["llm_enhanced 占比偏高，说明规则覆盖不足，优先补 procedure / project / meeting 类长尾表达。"],
        },
        "rows": [
            {
                "question": "最新制度是什么",
                "scenario": "policy_conflict",
                "model_route": "external_allowed",
                "analysis_source": "llm_enhanced",
                "analysis_confidence": 0.62,
                "analysis_reason": "问题包含版本变化语义",
                "query_scene": "policy_lookup",
                "data_classification": "internal",
                "matched_routes": ["original", "rewrite"],
                "metadata_boosted": True,
                "enterprise_entity_boosted": True,
                "enterprise_entity_matches": ["department"],
                "governance_boosted": True,
                "explainable_citation_count": 2,
                "refusal": False,
                "refusal_reason": "",
                "conflict_detected": True,
                "conflict_summary": "版本不一致",
                "expected_conflict": False,
                "faithfulness": 0.4,
            }
        ],
    }

    report = _render_explainability_report(payload)

    assert "# Eval Explainability Report" in report
    assert "analysis_source:heuristic" in report
    assert "Query Understanding Tuning" in report
    assert "matched_route:original" in report
    assert "enterprise_entity_matches" in report
    assert "最新制度是什么" in report
    assert "analysis_reason" in report
    assert "版本不一致" in report
