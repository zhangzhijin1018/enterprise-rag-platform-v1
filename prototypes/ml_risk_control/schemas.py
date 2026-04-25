"""风控原型共用数据结构。

这里把“训练样本结构”“在线推理输入”“ML 输出”“混合决策输出”统一定义在一起，
后续无论是训练脚本还是推理脚本，都会引用同一套结构，避免字段漂移。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


class RiskLevel(str, Enum):
    """统一的风险等级定义。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def normalize_risk_level(value: str | None) -> str:
    """把风险等级规整成 low / medium / high。

    这类规整逻辑一定要集中处理，避免不同模块各自实现导致行为不一致。
    """

    if not isinstance(value, str):
        return RiskLevel.LOW.value
    normalized = value.strip().lower()
    if normalized in _RISK_ORDER:
        return normalized
    return RiskLevel.LOW.value


def merge_risk_level(*levels: str | None) -> str:
    """返回多个风险等级中的最高风险等级。"""

    result = RiskLevel.LOW.value
    for level in levels:
        current = normalize_risk_level(level)
        if _RISK_ORDER[current] > _RISK_ORDER[result]:
            result = current
    return result


@dataclass
class NumericFeatureVector:
    """数值特征向量。

    这些特征故意设计成“业务语义非常清晰”的形式，便于：
    1. 训练时直接使用
    2. 推理时直接落审计
    3. 后续接入真实工程时可逐项映射
    """

    past_24h_query_count: float = 0.0
    high_risk_ratio_7d: float = 0.0
    failed_auth_count_7d: float = 0.0
    session_query_count: float = 0.0
    session_duration_sec: float = 0.0
    query_interval_sec: float = 60.0
    top1_score: float = 0.0
    top5_score_mean: float = 0.0
    restricted_hit_ratio: float = 0.0
    sensitive_hit_ratio: float = 0.0
    authority_score_mean: float = 0.0
    source_count: float = 1.0

    def ordered_names(self) -> list[str]:
        """返回稳定的特征名顺序。

        ONNX 推理阶段必须严格和训练阶段保持同一顺序，所以这里不允许依赖 dict 顺序猜测。
        """

        return list(asdict(self).keys())

    def to_list(self) -> list[float]:
        """按固定顺序转换为浮点列表。"""

        return [float(value) for value in asdict(self).values()]


@dataclass
class RiskSample:
    """训练样本结构。"""

    query_id: str
    query: str
    risk_label: str
    user_history: dict[str, Any] = field(default_factory=dict)
    session: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    label_reason: str | None = None


@dataclass
class RiskContext:
    """在线风控输入结构。

    这套结构有意对齐当前项目的 `RiskContext` 语义，但仍然保持独立，不直接依赖线上代码。
    """

    question: str
    user_id: str | None = None
    department: str | None = None
    role: str | None = None
    feature_vector: NumericFeatureVector = field(default_factory=NumericFeatureVector)


@dataclass
class MlRiskPrediction:
    """ML 模型输出。

    `risk_level_hint` 这个字段名是故意和现有项目的接入点对齐的。
    """

    risk_level_hint: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str
    model_version: str


@dataclass
class RuleDecision:
    """规则引擎输出。"""

    allow: bool
    action: str
    risk_level: str
    reason: str


@dataclass
class HybridRiskDecision:
    """混合风控最终输出。"""

    allow: bool
    action: str
    final_risk_level: str
    rule_risk_level: str
    ml_risk_level_hint: str
    rule_reason: str
    ml_confidence: float
    probabilities: dict[str, float]
