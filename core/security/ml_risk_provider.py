"""ML 风控 hint provider。

这层的职责非常克制：
1. 只负责根据输入上下文和特征给出 `risk_level_hint`
2. 不直接做 allow / deny / local_only / minimize 等最终安全动作
3. 失败时优先回退，不阻断现有规则风控主链路

当前第一轮接入只在 request-level 使用，因此这里优先实现：
- disabled provider：默认关闭
- mock provider：本地可跑、便于联调和测试
- onnx provider：为后续接真实模型保留正式入口
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from core.security.risk_engine import RiskContext

_FEATURE_ORDER = [
    "past_24h_query_count",
    "high_risk_ratio_7d",
    "failed_auth_count_7d",
    "session_query_count",
    "session_duration_sec",
    "query_interval_sec",
    "top1_score",
    "top5_score_mean",
    "restricted_hit_ratio",
    "sensitive_hit_ratio",
    "authority_score_mean",
    "source_count",
]

_HIGH_RISK_PATTERNS = (
    "导出全部",
    "完整名单",
    "全部名单",
    "身份证号",
    "银行卡",
    "账号口令",
    "薪资明细",
)


def _normalize_risk_level(value: object) -> str | None:
    """规整风险等级。"""

    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized
    return None


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """把输入稳妥转换为浮点数。"""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _nested_get(mapping: dict[str, Any], *path: str, default: Any = None) -> Any:
    """从嵌套 dict 中取值。"""

    current: Any = mapping
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


@dataclass(slots=True)
class MLRiskHintResult:
    """ML 风控 hint 输出。"""

    risk_level_hint: str | None = None
    confidence: float | None = None
    provider: str = "disabled"
    fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class MLRiskHintProvider(Protocol):
    """ML 风控 hint provider 协议。"""

    def predict(self, context: RiskContext, feature_bundle: dict[str, Any]) -> MLRiskHintResult:
        """根据上下文和特征输出风险等级提示。"""


class DisabledMLRiskHintProvider:
    """关闭状态下的 provider。"""

    def predict(self, context: RiskContext, feature_bundle: dict[str, Any]) -> MLRiskHintResult:
        _ = (context, feature_bundle)
        return MLRiskHintResult(provider="disabled")


class MockMLRiskHintProvider:
    """本地可跑的 mock provider。

    它不是正式 ML 模型，但具备两类价值：
    1. 第一轮接入时打通配置、装配、回退和审计链路
    2. 在没有 ONNX 依赖和模型产物时，仍能做联调与回归测试
    """

    def predict(self, context: RiskContext, feature_bundle: dict[str, Any]) -> MLRiskHintResult:
        question = context.question
        past_24h_query_count = _coerce_float(feature_bundle.get("past_24h_query_count"))
        high_risk_ratio_7d = _coerce_float(feature_bundle.get("high_risk_ratio_7d"))
        failed_auth_count_7d = _coerce_float(feature_bundle.get("failed_auth_count_7d"))
        session_query_count = _coerce_float(feature_bundle.get("session_query_count"))
        query_interval_sec = _coerce_float(feature_bundle.get("query_interval_sec"), default=60.0)

        strong_signals: list[str] = []
        medium_signals: list[str] = []

        if any(pattern in question for pattern in _HIGH_RISK_PATTERNS):
            strong_signals.append("text_pattern")
        if failed_auth_count_7d >= 3:
            strong_signals.append("failed_auth_spike")
        if high_risk_ratio_7d >= 0.40:
            strong_signals.append("high_risk_history_ratio")
        if past_24h_query_count >= 30:
            strong_signals.append("query_volume_spike")
        if session_query_count >= 8 and query_interval_sec <= 10:
            strong_signals.append("burst_session_pattern")

        if high_risk_ratio_7d >= 0.15:
            medium_signals.append("elevated_history_ratio")
        if past_24h_query_count >= 12:
            medium_signals.append("elevated_query_volume")
        if session_query_count >= 5:
            medium_signals.append("elevated_session_depth")
        if query_interval_sec <= 20:
            medium_signals.append("short_query_interval")

        if strong_signals:
            return MLRiskHintResult(
                risk_level_hint="high",
                confidence=0.88,
                provider="mock",
                metadata={"signals": strong_signals, "feature_bundle": feature_bundle},
            )
        if medium_signals:
            return MLRiskHintResult(
                risk_level_hint="medium",
                confidence=0.67,
                provider="mock",
                metadata={"signals": medium_signals, "feature_bundle": feature_bundle},
            )
        return MLRiskHintResult(
            risk_level_hint="low",
            confidence=0.56,
            provider="mock",
            metadata={"signals": [], "feature_bundle": feature_bundle},
        )


class OnnxMLRiskHintProvider:
    """基于 ONNX Runtime 的正式 provider。

    当前第一轮只要求把入口打通，因此这里采用延迟加载：
    - 初始化阶段不强依赖 onnxruntime / transformers / numpy
    - 真正执行 predict 时再加载
    - 失败后由 `safe_predict_ml_risk_hint` 决定是否回退
    """

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._loaded = False
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._np: Any | None = None
        self._manifest: dict[str, Any] = {}
        self._feature_stats: dict[str, dict[str, float]] = {}
        self._id_to_label: dict[int, str] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        try:
            import numpy as np
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("onnx 风控 provider 依赖缺失") from exc

        model_dir = Path(getattr(self.settings, "ml_risk_model_dir", "./modes/ml-risk"))
        onnx_path = Path(getattr(self.settings, "ml_risk_onnx_path", str(model_dir / "risk_classifier.onnx")))
        manifest_path = model_dir / "training_manifest.json"
        feature_stats_path = model_dir / "feature_stats.json"
        label_to_id_path = model_dir / "label_to_id.json"
        tokenizer_dir = model_dir / "tokenizer"

        if not manifest_path.exists():
            raise FileNotFoundError(f"未找到 ML 风控 manifest: {manifest_path}")
        if not feature_stats_path.exists():
            raise FileNotFoundError(f"未找到 ML 风控 feature_stats: {feature_stats_path}")
        if not label_to_id_path.exists():
            raise FileNotFoundError(f"未找到 ML 风控标签映射: {label_to_id_path}")
        if not onnx_path.exists():
            raise FileNotFoundError(f"未找到 ML 风控 ONNX 模型: {onnx_path}")

        import json

        self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._feature_stats = json.loads(feature_stats_path.read_text(encoding="utf-8"))
        label_to_id = json.loads(label_to_id_path.read_text(encoding="utf-8"))
        self._id_to_label = {int(value): key for key, value in label_to_id.items()}
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self._np = np
        self._loaded = True

    def _normalize_features(self, feature_bundle: dict[str, Any]) -> list[float]:
        normalized: list[float] = []
        for feature_name in _FEATURE_ORDER:
            value = _coerce_float(feature_bundle.get(feature_name), default=0.0)
            stat = self._feature_stats.get(feature_name, {"mean": 0.0, "std": 1.0})
            mean_value = _coerce_float(stat.get("mean"), default=0.0)
            std_value = _coerce_float(stat.get("std"), default=1.0)
            if std_value <= 1e-6:
                std_value = 1.0
            normalized.append((value - mean_value) / std_value)
        return normalized

    def predict(self, context: RiskContext, feature_bundle: dict[str, Any]) -> MLRiskHintResult:
        self._ensure_loaded()
        assert self._session is not None
        assert self._tokenizer is not None
        assert self._np is not None

        tokenized = self._tokenizer(
            context.question,
            truncation=True,
            padding="max_length",
            max_length=int(self._manifest.get("max_length", 128)),
            return_tensors="np",
        )
        dense_features = self._np.array(
            [self._normalize_features(feature_bundle)],
            dtype=self._np.float32,
        )
        logits = self._session.run(
            ["logits"],
            {
                "input_ids": tokenized["input_ids"].astype(self._np.int64),
                "attention_mask": tokenized["attention_mask"].astype(self._np.int64),
                "dense_features": dense_features,
            },
        )[0]
        shifted = logits - self._np.max(logits, axis=-1, keepdims=True)
        exp_values = self._np.exp(shifted)
        probabilities = exp_values / self._np.sum(exp_values, axis=-1, keepdims=True)
        predicted_index = int(self._np.argmax(probabilities[0]))
        risk_level_hint = self._id_to_label.get(predicted_index)
        return MLRiskHintResult(
            risk_level_hint=_normalize_risk_level(risk_level_hint),
            confidence=float(probabilities[0][predicted_index]),
            provider="onnx",
            metadata={
                "feature_bundle": feature_bundle,
                "probabilities": {
                    self._id_to_label.get(index, str(index)): float(probabilities[0][index])
                    for index in range(len(probabilities[0]))
                },
            },
        )


def build_request_risk_feature_bundle(*, question: str, user_context: dict[str, Any]) -> dict[str, Any]:
    """构造 request-level 风控特征。

    第一轮只使用请求入口已有的上下文，不新增 API 字段。
    `session_metadata` 兼容两种写法：
    - 扁平字段：`session_metadata["past_24h_query_count"]`
    - 分组字段：`session_metadata["user_history"]["past_24h_query_count"]`
    """

    session_metadata = user_context.get("session_metadata") or {}
    feature_bundle = {
        "past_24h_query_count": _coerce_float(
            session_metadata.get("past_24h_query_count"),
            default=_coerce_float(_nested_get(session_metadata, "user_history", "past_24h_query_count")),
        ),
        "high_risk_ratio_7d": _coerce_float(
            session_metadata.get("high_risk_ratio_7d"),
            default=_coerce_float(_nested_get(session_metadata, "user_history", "high_risk_ratio_7d")),
        ),
        "failed_auth_count_7d": _coerce_float(
            session_metadata.get("failed_auth_count_7d"),
            default=_coerce_float(_nested_get(session_metadata, "user_history", "failed_auth_count_7d")),
        ),
        "session_query_count": _coerce_float(
            session_metadata.get("session_query_count"),
            default=_coerce_float(_nested_get(session_metadata, "session", "session_query_count")),
        ),
        "session_duration_sec": _coerce_float(
            session_metadata.get("session_duration_sec"),
            default=_coerce_float(_nested_get(session_metadata, "session", "session_duration_sec")),
        ),
        "query_interval_sec": _coerce_float(
            session_metadata.get("query_interval_sec"),
            default=_coerce_float(_nested_get(session_metadata, "session", "query_interval_sec"), default=60.0),
        ),
        # 第一轮 request-level 先不依赖 retrieval 分布特征，统一补默认值。
        "top1_score": 0.0,
        "top5_score_mean": 0.0,
        "restricted_hit_ratio": 0.0,
        "sensitive_hit_ratio": 0.0,
        "authority_score_mean": 0.0,
        "source_count": 1.0,
        # 这些字段主要用于 mock provider 辅助判断或后续审计。
        "question_length": len(question),
        "has_enterprise_context": any(
            [
                user_context.get("user_id"),
                user_context.get("department"),
                user_context.get("role"),
                user_context.get("project_ids"),
                user_context.get("clearance_level"),
            ]
        ),
    }
    return feature_bundle


def build_ml_risk_provider(settings: Any) -> MLRiskHintProvider:
    """根据配置构造 ML 风控 provider。"""

    if not bool(getattr(settings, "enable_ml_risk_hint", False)):
        return DisabledMLRiskHintProvider()

    provider_name = str(getattr(settings, "ml_risk_hint_provider", "disabled") or "disabled").strip().lower()
    if provider_name == "mock":
        return MockMLRiskHintProvider()
    if provider_name == "onnx":
        return OnnxMLRiskHintProvider(settings)
    return DisabledMLRiskHintProvider()


def safe_predict_ml_risk_hint(
    provider: MLRiskHintProvider | None,
    *,
    context: RiskContext,
    feature_bundle: dict[str, Any],
    settings: Any,
) -> MLRiskHintResult:
    """安全执行 ML 风控 hint 推理。"""

    if not bool(getattr(settings, "enable_ml_risk_hint", False)):
        return MLRiskHintResult(provider="disabled")
    if not bool(getattr(settings, "ml_risk_request_stage_enabled", True)) and context.stage == "request":
        return MLRiskHintResult(provider="disabled")
    if provider is None:
        return MLRiskHintResult(provider="disabled")

    try:
        result = provider.predict(context, feature_bundle)
        normalized_hint = _normalize_risk_level(result.risk_level_hint)
        return MLRiskHintResult(
            risk_level_hint=normalized_hint,
            confidence=result.confidence,
            provider=result.provider,
            fallback=result.fallback,
            metadata=result.metadata,
        )
    except Exception as exc:
        if bool(getattr(settings, "ml_risk_fail_open", True)):
            return MLRiskHintResult(
                provider="fallback",
                fallback=True,
                metadata={"error": type(exc).__name__, "message": str(exc)},
            )
        raise

