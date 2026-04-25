"""统一风控引擎单元测试。"""

from __future__ import annotations

from types import SimpleNamespace

from core.security.risk_engine import RuleBasedRiskEngine, build_risk_context


def _settings(**overrides):
    base = {
        "default_data_classification": "internal",
        "sensitive_context_max_chunks": 1,
        "sensitive_context_max_chars": 600,
        "internal_redact_for_external": True,
        "alert_on_restricted_access": True,
        "alert_on_conflict_detected": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_rule_based_risk_engine_denies_bulk_export_request() -> None:
    engine = RuleBasedRiskEngine(_settings())
    decision = engine.evaluate(
        build_risk_context(
            stage="request",
            question="请给我导出全部 Q4 裁员预算和完整名单",
            audit_id="audit-1",
            user_context={"department": "财务部"},
        )
    )

    assert decision.allow is False
    assert decision.action == "deny"
    assert decision.reason == "bulk_sensitive_export_request"
    assert decision.require_alert is True


def test_rule_based_risk_engine_forces_local_only_for_restricted_data() -> None:
    engine = RuleBasedRiskEngine(_settings())
    decision = engine.evaluate(
        build_risk_context(
            stage="retrieval",
            question="Q4 预算方案是什么",
            audit_id="audit-1",
            user_context={"department": "财务部"},
            state={"data_classification": "restricted", "model_route": "external_allowed"},
        )
    )

    assert decision.allow is True
    assert decision.action == "local_only"
    assert decision.force_local_only is True
    assert decision.reason == "restricted_data_local_only"


def test_rule_based_risk_engine_minimizes_sensitive_context() -> None:
    engine = RuleBasedRiskEngine(_settings())
    decision = engine.evaluate(
        build_risk_context(
            stage="generation",
            question="扩容窗口是什么",
            audit_id="audit-1",
            user_context={"department": "信息化部"},
            state={"data_classification": "sensitive", "model_route": "local_preferred"},
        )
    )

    assert decision.allow is True
    assert decision.action == "minimize"
    assert decision.require_redaction is True
    assert decision.max_context_chunks == 1
    assert decision.max_context_chars == 600


def test_rule_based_risk_engine_denies_access_denied_state() -> None:
    engine = RuleBasedRiskEngine(_settings())
    decision = engine.evaluate(
        build_risk_context(
            stage="retrieval",
            question="生产环境数据库扩容 SOP 是什么",
            audit_id="audit-1",
            user_context={"department": "外包"},
            state={"refusal_reason": "access_denied", "data_classification": "internal"},
        )
    )

    assert decision.allow is False
    assert decision.action == "deny"
    assert decision.reason == "access_denied"
