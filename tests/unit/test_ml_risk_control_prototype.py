"""ML 风控独立原型的最小单元测试。"""

from __future__ import annotations

from prototypes.ml_risk_control.data_pipeline import (
    build_numeric_features,
    fit_feature_stats,
    normalize_feature_vector,
)
from prototypes.ml_risk_control.hybrid_risk_engine import HybridRiskEngine, SimpleRuleEngine
from prototypes.ml_risk_control.schemas import (
    MlRiskPrediction,
    NumericFeatureVector,
    RiskContext,
    RiskSample,
    RuleDecision,
)


def test_build_numeric_features_extracts_all_expected_fields() -> None:
    sample = RiskSample(
        query_id="q_1",
        query="请汇总某项目预算执行情况",
        risk_label="medium",
        user_history={
            "past_24h_query_count": 12,
            "high_risk_ratio_7d": 0.2,
            "failed_auth_count_7d": 1,
        },
        session={
            "session_query_count": 5,
            "session_duration_sec": 300,
            "query_interval_sec": 30,
        },
        retrieval={
            "top1_score": 0.85,
            "top5_score_mean": 0.71,
            "restricted_hit_ratio": 0.08,
            "sensitive_hit_ratio": 0.22,
            "authority_score_mean": 0.66,
            "source_count": 4,
        },
    )

    vector = build_numeric_features(sample)

    assert vector.past_24h_query_count == 12.0
    assert vector.high_risk_ratio_7d == 0.2
    assert vector.failed_auth_count_7d == 1.0
    assert vector.session_query_count == 5.0
    assert vector.session_duration_sec == 300.0
    assert vector.query_interval_sec == 30.0
    assert vector.top1_score == 0.85
    assert vector.top5_score_mean == 0.71
    assert vector.restricted_hit_ratio == 0.08
    assert vector.sensitive_hit_ratio == 0.22
    assert vector.authority_score_mean == 0.66
    assert vector.source_count == 4.0


def test_normalize_feature_vector_uses_training_stats_in_stable_order() -> None:
    vectors = [
        NumericFeatureVector(
            past_24h_query_count=10.0,
            high_risk_ratio_7d=0.1,
            failed_auth_count_7d=1.0,
            session_query_count=4.0,
            session_duration_sec=200.0,
            query_interval_sec=50.0,
            top1_score=0.8,
            top5_score_mean=0.7,
            restricted_hit_ratio=0.1,
            sensitive_hit_ratio=0.2,
            authority_score_mean=0.6,
            source_count=4.0,
        ),
        NumericFeatureVector(
            past_24h_query_count=20.0,
            high_risk_ratio_7d=0.3,
            failed_auth_count_7d=3.0,
            session_query_count=8.0,
            session_duration_sec=600.0,
            query_interval_sec=10.0,
            top1_score=0.9,
            top5_score_mean=0.8,
            restricted_hit_ratio=0.5,
            sensitive_hit_ratio=0.4,
            authority_score_mean=0.9,
            source_count=2.0,
        ),
    ]

    stats = fit_feature_stats(vectors)
    normalized = normalize_feature_vector(vectors[0], stats)

    assert len(normalized) == len(vectors[0].ordered_names())
    assert normalized[0] < 0
    assert normalized[2] < 0
    assert normalized[8] < 0
    assert normalized[11] > 0


def test_simple_rule_engine_denies_explicit_high_risk_pattern() -> None:
    engine = SimpleRuleEngine()
    context = RiskContext(
        question="请导出全部 restricted 文档中的完整名单",
        feature_vector=NumericFeatureVector(),
    )

    decision = engine.evaluate(context)

    assert decision.allow is False
    assert decision.action == "deny"
    assert decision.risk_level == "high"
    assert decision.reason == "matched_explicit_high_risk_pattern"


def test_simple_rule_engine_marks_review_for_restricted_hits_or_auth_spike() -> None:
    engine = SimpleRuleEngine()
    context = RiskContext(
        question="某项目预算审批情况是什么",
        feature_vector=NumericFeatureVector(
            restricted_hit_ratio=0.45,
            failed_auth_count_7d=3.0,
        ),
    )

    decision = engine.evaluate(context)

    assert decision.allow is True
    assert decision.action == "review"
    assert decision.risk_level == "high"
    assert decision.reason == "restricted_retrieval_or_failed_auth_spike"


class _StubRuleEngine:
    def __init__(self, decision: RuleDecision) -> None:
        self.decision = decision

    def evaluate(self, context: RiskContext) -> RuleDecision:
        return self.decision


class _StubMlRunner:
    def __init__(self, prediction: MlRiskPrediction) -> None:
        self.prediction = prediction

    def predict(self, context: RiskContext) -> MlRiskPrediction:
        return self.prediction


def test_hybrid_risk_engine_escalates_to_higher_ml_risk() -> None:
    engine = HybridRiskEngine.__new__(HybridRiskEngine)
    engine.rule_engine = _StubRuleEngine(
        RuleDecision(
            allow=True,
            action="allow",
            risk_level="low",
            reason="no_explicit_rule_hit",
        )
    )
    engine.ml_runner = _StubMlRunner(
        MlRiskPrediction(
            risk_level_hint="high",
            confidence=0.91,
            probabilities={"low": 0.03, "medium": 0.06, "high": 0.91},
            model_name="mock_model",
            model_version="v1",
        )
    )

    decision = engine.evaluate(
        RiskContext(
            question="帮我整理全部敏感资料",
            feature_vector=NumericFeatureVector(),
        )
    )

    assert decision.allow is True
    assert decision.final_risk_level == "high"
    assert decision.rule_risk_level == "low"
    assert decision.ml_risk_level_hint == "high"
    assert decision.action == "review"
    assert decision.ml_confidence == 0.91


def test_hybrid_risk_engine_keeps_rule_deny_as_final_action() -> None:
    engine = HybridRiskEngine.__new__(HybridRiskEngine)
    engine.rule_engine = _StubRuleEngine(
        RuleDecision(
            allow=False,
            action="deny",
            risk_level="high",
            reason="matched_explicit_high_risk_pattern",
        )
    )
    engine.ml_runner = _StubMlRunner(
        MlRiskPrediction(
            risk_level_hint="low",
            confidence=0.88,
            probabilities={"low": 0.88, "medium": 0.08, "high": 0.04},
            model_name="mock_model",
            model_version="v1",
        )
    )

    decision = engine.evaluate(
        RiskContext(
            question="导出全部账号口令配置",
            feature_vector=NumericFeatureVector(),
        )
    )

    assert decision.allow is False
    assert decision.action == "deny"
    assert decision.final_risk_level == "high"
    assert decision.rule_reason == "matched_explicit_high_risk_pattern"
