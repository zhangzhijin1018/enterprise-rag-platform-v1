"""规则 + ML 混合风控推理。

这个模块包含三层职责：
1. ONNX Runtime 推理，得到 `risk_level_hint`
2. 规则引擎显式兜底
3. 冲突时取高风险，形成最终决策
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .data_pipeline import normalize_feature_vector
from .schemas import (
    HybridRiskDecision,
    MlRiskPrediction,
    NumericFeatureVector,
    RiskContext,
    RuleDecision,
    merge_risk_level,
)

HIGH_RISK_PATTERNS = (
    "导出全部",
    "完整名单",
    "身份证号",
    "银行卡",
    "restricted",
    "账号口令",
    "薪资明细",
)


def _require_inference_dependencies() -> dict[str, Any]:
    """延迟导入推理依赖。"""

    try:
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("推理依赖缺失，请先安装 numpy / onnxruntime / transformers") from exc
    return {"np": np, "ort": ort, "AutoTokenizer": AutoTokenizer}


class SimpleRuleEngine:
    """最小可解释规则引擎。

    规则仍然是最终裁决器，因此这里保留 deny / review / allow 的控制权。
    """

    def evaluate(self, context: RiskContext) -> RuleDecision:
        question = context.question
        vector = context.feature_vector

        if any(pattern in question for pattern in HIGH_RISK_PATTERNS):
            return RuleDecision(
                allow=False,
                action="deny",
                risk_level="high",
                reason="matched_explicit_high_risk_pattern",
            )

        if vector.restricted_hit_ratio >= 0.4 or vector.failed_auth_count_7d >= 3:
            return RuleDecision(
                allow=True,
                action="review",
                risk_level="high",
                reason="restricted_retrieval_or_failed_auth_spike",
            )

        if vector.sensitive_hit_ratio >= 0.2 or vector.high_risk_ratio_7d >= 0.25:
            return RuleDecision(
                allow=True,
                action="review",
                risk_level="medium",
                reason="sensitive_hit_ratio_or_history_risk_elevated",
            )

        return RuleDecision(
            allow=True,
            action="allow",
            risk_level="low",
            reason="no_explicit_rule_hit",
        )


class OnnxRiskModelRunner:
    """ONNX 风控分类模型运行器。"""

    def __init__(self, classifier_dir: str | Path, onnx_path: str | Path) -> None:
        deps = _require_inference_dependencies()
        self.np = deps["np"]
        ort = deps["ort"]
        AutoTokenizer = deps["AutoTokenizer"]

        classifier_path = Path(classifier_dir)
        self.manifest = json.loads((classifier_path / "training_manifest.json").read_text(encoding="utf-8"))
        self.feature_stats = json.loads((classifier_path / "feature_stats.json").read_text(encoding="utf-8"))
        self.label_to_id = json.loads((classifier_path / "label_to_id.json").read_text(encoding="utf-8"))
        self.id_to_label = {int(value): key for key, value in self.label_to_id.items()}

        tokenizer_dir = classifier_path / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    @staticmethod
    def _softmax(self, logits: Any) -> Any:
        """对 logits 做 softmax。"""

        shifted = logits - self.np.max(logits, axis=-1, keepdims=True)
        exp_values = self.np.exp(shifted)
        return exp_values / self.np.sum(exp_values, axis=-1, keepdims=True)

    def predict(self, context: RiskContext) -> MlRiskPrediction:
        """输出 ML 风险提示。

        这是未来真正接入主项目时最需要保留的接口语义。
        """

        tokenized = self.tokenizer(
            context.question,
            truncation=True,
            padding="max_length",
            max_length=int(self.manifest["max_length"]),
            return_tensors="np",
        )
        dense_features = self.np.array(
            [normalize_feature_vector(context.feature_vector, self.feature_stats)],
            dtype=self.np.float32,
        )
        logits = self.session.run(
            ["logits"],
            {
                "input_ids": tokenized["input_ids"].astype(self.np.int64),
                "attention_mask": tokenized["attention_mask"].astype(self.np.int64),
                "dense_features": dense_features,
            },
        )[0]
        probabilities = self._softmax(logits)[0]
        predicted_index = int(self.np.argmax(probabilities))
        risk_level_hint = self.id_to_label[predicted_index]
        probability_map = {
            self.id_to_label[index]: round(float(probabilities[index]), 6)
            for index in range(len(probabilities))
        }
        return MlRiskPrediction(
            risk_level_hint=risk_level_hint,
            confidence=round(float(probabilities[predicted_index]), 6),
            probabilities=probability_map,
            model_name="bert_dense_feature_risk_classifier",
            model_version=str(self.manifest.get("best_accuracy", "unknown")),
        )


class HybridRiskEngine:
    """规则 + ONNX 模型混合判定引擎。"""

    def __init__(self, classifier_dir: str | Path, onnx_path: str | Path) -> None:
        self.rule_engine = SimpleRuleEngine()
        self.ml_runner = OnnxRiskModelRunner(classifier_dir=classifier_dir, onnx_path=onnx_path)

    def evaluate(self, context: RiskContext) -> HybridRiskDecision:
        """输出最终混合决策。

        冲突策略：
        - 如果规则直接 deny，最终直接 deny
        - 否则取规则和 ML 中更高的风险等级
        - 当 ML 高于规则时，最终风险升级
        """

        rule_decision = self.rule_engine.evaluate(context)
        ml_prediction = self.ml_runner.predict(context)

        final_risk_level = merge_risk_level(rule_decision.risk_level, ml_prediction.risk_level_hint)
        final_allow = rule_decision.allow
        final_action = rule_decision.action

        if final_allow and final_action == "allow" and final_risk_level == "high":
            # 即便规则没有直接 deny，只要 ML 把风险顶到 high，也不要继续按纯 allow 放行。
            final_action = "review"

        return HybridRiskDecision(
            allow=final_allow,
            action=final_action,
            final_risk_level=final_risk_level,
            rule_risk_level=rule_decision.risk_level,
            ml_risk_level_hint=ml_prediction.risk_level_hint,
            rule_reason=rule_decision.reason,
            ml_confidence=ml_prediction.confidence,
            probabilities=ml_prediction.probabilities,
        )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="执行规则 + ML 混合风控推理")
    parser.add_argument("--classifier-dir", type=str, required=True, help="分类模型目录")
    parser.add_argument("--onnx-path", type=str, required=True, help="ONNX 文件路径")
    parser.add_argument("--question", type=str, required=True, help="待评估查询")
    parser.add_argument("--user-id", type=str, default="", help="用户 ID")
    parser.add_argument("--department", type=str, default="", help="部门")
    parser.add_argument("--role", type=str, default="", help="角色")
    parser.add_argument("--past-24h-query-count", type=float, default=0.0)
    parser.add_argument("--high-risk-ratio-7d", type=float, default=0.0)
    parser.add_argument("--failed-auth-count-7d", type=float, default=0.0)
    parser.add_argument("--session-query-count", type=float, default=0.0)
    parser.add_argument("--session-duration-sec", type=float, default=0.0)
    parser.add_argument("--query-interval-sec", type=float, default=60.0)
    parser.add_argument("--top1-score", type=float, default=0.0)
    parser.add_argument("--top5-score-mean", type=float, default=0.0)
    parser.add_argument("--restricted-hit-ratio", type=float, default=0.0)
    parser.add_argument("--sensitive-hit-ratio", type=float, default=0.0)
    parser.add_argument("--authority-score-mean", type=float, default=0.0)
    parser.add_argument("--source-count", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    """命令行推理入口。"""

    args = parse_args()
    feature_vector = NumericFeatureVector(
        past_24h_query_count=args.past_24h_query_count,
        high_risk_ratio_7d=args.high_risk_ratio_7d,
        failed_auth_count_7d=args.failed_auth_count_7d,
        session_query_count=args.session_query_count,
        session_duration_sec=args.session_duration_sec,
        query_interval_sec=args.query_interval_sec,
        top1_score=args.top1_score,
        top5_score_mean=args.top5_score_mean,
        restricted_hit_ratio=args.restricted_hit_ratio,
        sensitive_hit_ratio=args.sensitive_hit_ratio,
        authority_score_mean=args.authority_score_mean,
        source_count=args.source_count,
    )
    context = RiskContext(
        question=args.question,
        user_id=args.user_id or None,
        department=args.department or None,
        role=args.role or None,
        feature_vector=feature_vector,
    )
    engine = HybridRiskEngine(classifier_dir=args.classifier_dir, onnx_path=args.onnx_path)
    decision = engine.evaluate(context)
    print(
        json.dumps(
            {
                "context": {
                    "question": context.question,
                    "user_id": context.user_id,
                    "department": context.department,
                    "role": context.role,
                    "feature_vector": asdict(context.feature_vector),
                },
                "decision": asdict(decision),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
