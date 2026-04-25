"""审计与高风险日志单元测试。"""

from __future__ import annotations

from types import SimpleNamespace

from core.observability.audit import assess_query_risk, build_audit_event, should_trigger_alert


def _settings(**overrides):
    base = {
        "audit_log_enabled": True,
        "audit_log_redact_content": True,
        "audit_log_preview_chars": 120,
        "alert_on_high_risk_queries": True,
        "alert_on_restricted_access": True,
        "alert_on_conflict_detected": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_assess_query_risk_marks_sensitive_financial_questions_as_high() -> None:
    risk = assess_query_risk("请给我导出全部 Q4 裁员预算和完整名单")
    assert risk == "high"


def test_build_audit_event_redacts_prompt_and_output_preview() -> None:
    event = build_audit_event(
        stage="prompt_audited",
        audit_id="audit-1",
        question="联系张三，手机号 13812345678 是多少？",
        user_context={"department": "信息化部", "role": "engineer"},
        settings=_settings(),
        state={
            "data_classification": "internal",
            "model_route": "external_allowed",
            "reranked_hits": [
                {"chunk_id": "c1", "metadata": {"doc_id": "doc-1"}},
            ],
        },
        prompt="手机号 13812345678 邮箱 test@example.com",
        output="联系人 test@example.com",
    )

    assert event["risk_level"] in {"medium", "high"}
    assert event["question_hash"]
    assert event["prompt_hash"]
    assert event["output_hash"]
    assert "13812345678" not in (event["question_preview"] or "")
    assert "13812345678" not in (event["prompt_preview"] or "")
    assert "test@example.com" not in (event["output_preview"] or "")
    assert event["retrieved_chunk_ids"] == ["c1"]
    assert event["retrieved_doc_ids"] == ["doc-1"]


def test_should_trigger_alert_for_high_risk_and_restricted_requests() -> None:
    settings = _settings()
    assert should_trigger_alert(
        settings=settings,
        state={"data_classification": "internal"},
        risk_level="high",
    )
    assert should_trigger_alert(
        settings=settings,
        state={"data_classification": "restricted"},
        risk_level="low",
    )
    assert should_trigger_alert(
        settings=settings,
        state={"conflict_detected": True},
        risk_level="low",
    )
