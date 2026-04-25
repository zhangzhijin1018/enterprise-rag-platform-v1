"""ML 风控 hint provider 单元测试。"""

from __future__ import annotations

from types import SimpleNamespace

from core.security.ml_risk_provider import (
    MLRiskHintResult,
    MockMLRiskHintProvider,
    build_ml_risk_provider,
    build_request_risk_feature_bundle,
    safe_predict_ml_risk_hint,
)
from core.security.risk_engine import RiskContext


def _settings(**overrides):
    base = {
        "enable_ml_risk_hint": True,
        "ml_risk_hint_provider": "mock",
        "ml_risk_fail_open": True,
        "ml_risk_request_stage_enabled": True,
        "ml_risk_model_dir": "./modes/ml-risk",
        "ml_risk_onnx_path": "./modes/ml-risk/risk_classifier.onnx",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_request_risk_feature_bundle_supports_flat_and_nested_session_metadata() -> None:
    feature_bundle = build_request_risk_feature_bundle(
        question="帮我汇总预算执行情况",
        user_context={
            "user_id": "u_1",
            "department": "财务部",
            "session_metadata": {
                "past_24h_query_count": 18,
                "user_history": {"high_risk_ratio_7d": 0.25, "failed_auth_count_7d": 2},
                "session": {"session_query_count": 6, "session_duration_sec": 400, "query_interval_sec": 12},
            },
        },
    )

    assert feature_bundle["past_24h_query_count"] == 18.0
    assert feature_bundle["high_risk_ratio_7d"] == 0.25
    assert feature_bundle["failed_auth_count_7d"] == 2.0
    assert feature_bundle["session_query_count"] == 6.0
    assert feature_bundle["session_duration_sec"] == 400.0
    assert feature_bundle["query_interval_sec"] == 12.0
    assert feature_bundle["has_enterprise_context"] is True
    assert feature_bundle["question_length"] > 0


def test_mock_ml_risk_provider_returns_high_for_burst_and_auth_spike() -> None:
    provider = MockMLRiskHintProvider()
    result = provider.predict(
        RiskContext(stage="request", audit_id="audit-1", question="请帮我整理相关资料"),
        {
            "past_24h_query_count": 35,
            "high_risk_ratio_7d": 0.45,
            "failed_auth_count_7d": 3,
            "session_query_count": 10,
            "query_interval_sec": 8,
        },
    )

    assert result.risk_level_hint == "high"
    assert result.provider == "mock"
    assert result.confidence == 0.88


def test_safe_predict_ml_risk_hint_falls_back_when_provider_raises() -> None:
    class _BrokenProvider:
        def predict(self, context, feature_bundle):  # noqa: ANN001
            _ = (context, feature_bundle)
            raise RuntimeError("provider broken")

    result = safe_predict_ml_risk_hint(
        _BrokenProvider(),
        context=RiskContext(stage="request", audit_id="audit-1", question="hello"),
        feature_bundle={},
        settings=_settings(ml_risk_fail_open=True),
    )

    assert result.provider == "fallback"
    assert result.fallback is True
    assert result.risk_level_hint is None
    assert result.metadata["error"] == "RuntimeError"


def test_build_ml_risk_provider_returns_disabled_when_feature_closed() -> None:
    provider = build_ml_risk_provider(_settings(enable_ml_risk_hint=False, ml_risk_hint_provider="disabled"))
    result = provider.predict(
        RiskContext(stage="request", audit_id="audit-1", question="hello"),
        {},
    )

    assert isinstance(result, MLRiskHintResult)
    assert result.provider == "disabled"
