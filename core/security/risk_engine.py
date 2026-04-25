"""统一风控引擎抽象与本地规则实现。

这层的职责不是直接做检索或生成，而是：
1. 把请求、检索、生成三个阶段统一抽象成可评估的 RiskContext
2. 输出结构化的 RiskDecision
3. 让上层主链路根据决策执行 allow / redact / minimize / local_only / deny
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from core.observability.audit import assess_query_risk
from core.retrieval.access_control import normalize_classification

_RISK_LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}
_BULK_EXPORT_PATTERNS = (
    "导出全部",
    "完整名单",
    "全部名单",
    "原文全文",
    "全部预算明细",
    "所有人员信息",
)


def _normalize_risk_level(value: object) -> str:
    """把风险等级规整为 low / medium / high。"""

    if not isinstance(value, str):
        return "low"
    level = value.strip().lower()
    if level in _RISK_LEVEL_ORDER:
        return level
    return "low"


def _merge_risk_level(*levels: object) -> str:
    """返回多个风险等级中的最高值。"""

    highest = "low"
    for level in levels:
        normalized = _normalize_risk_level(level)
        if _RISK_LEVEL_ORDER[normalized] > _RISK_LEVEL_ORDER[highest]:
            highest = normalized
    return highest


@dataclass(slots=True)
class RiskContext:
    """风控决策输入。

    这里尽量只保留“做决策真正需要的最小上下文”，方便后续：
    - 接本地规则引擎
    - 接远程 PDP / OPA / 企业风控中心
    """

    stage: str
    audit_id: str
    question: str
    user_id: str | None = None
    department: str | None = None
    role: str | None = None
    project_ids: list[str] = field(default_factory=list)
    query_scene: str | None = None
    allow_external_llm: bool | None = None
    data_classification: str | None = None
    model_route: str | None = None
    refusal_reason: str | None = None
    matched_doc_ids: list[str] = field(default_factory=list)
    matched_chunk_ids: list[str] = field(default_factory=list)
    conflict_detected: bool = False
    risk_level_hint: str | None = None


@dataclass(slots=True)
class RiskDecision:
    """风控决策输出。

    它表达的不是“模型怎么回答”，而是：
    - 这次请求允不允许继续
    - 允许的话，需要以什么安全策略继续
    """

    allow: bool = True
    action: str = "allow"
    risk_level: str = "low"
    reason: str = ""
    require_redaction: bool = False
    max_context_chunks: int | None = None
    max_context_chars: int | None = None
    force_local_only: bool = False
    require_alert: bool = False
    require_human_review: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class RiskEngine(Protocol):
    """风控引擎协议。

    当前项目默认实现是 `RuleBasedRiskEngine`；
    后续如果切到远程风控服务，只要保持这个接口不变即可。
    """

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """根据上下文输出风控决策。"""


def safe_evaluate_risk(engine: RiskEngine, context: RiskContext, settings: Any) -> RiskDecision:
    """带 fail-open / fail-close 语义地执行风控评估。

    设计目的：
    - 风控引擎正常时，按风控结果执行
    - 风控引擎异常时，依据配置决定是“保守拒绝”还是“允许主链路继续”
    """

    if not bool(getattr(settings, "enable_risk_engine", True)):
        return RiskDecision(
            allow=True,
            action="allow",
            risk_level=_merge_risk_level(context.risk_level_hint, assess_query_risk(context.question)),
            reason="risk_engine_disabled",
        )
    try:
        return engine.evaluate(context)
    except Exception:
        fail_open = bool(getattr(settings, "risk_engine_fail_open", True))
        return RiskDecision(
            allow=fail_open,
            action="allow" if fail_open else "deny",
            risk_level="high",
            reason="risk_engine_fail_open" if fail_open else "risk_engine_failure",
            require_alert=True,
        )


def build_risk_context(
    *,
    stage: str,
    question: str,
    audit_id: str,
    user_context: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> RiskContext:
    """从请求上下文和图状态组装统一风控输入。

    这一步负责把上游分散在：
    - ChatRequest
    - RAGState
    - reranked hits
    里的信息收敛成风控可直接消费的一份结构化对象。
    """

    user_context = user_context or {}
    state = state or {}
    reranked_hits = state.get("reranked_hits") or []
    return RiskContext(
        stage=stage,
        audit_id=audit_id,
        question=question,
        user_id=user_context.get("user_id"),
        department=user_context.get("department"),
        role=user_context.get("role"),
        project_ids=list(user_context.get("project_ids") or []),
        query_scene=user_context.get("query_scene"),
        allow_external_llm=user_context.get("allow_external_llm"),
        data_classification=state.get("data_classification"),
        model_route=state.get("model_route"),
        refusal_reason=state.get("refusal_reason"),
        matched_doc_ids=[
            (item.get("metadata") or {}).get("doc_id")
            for item in reranked_hits
            if (item.get("metadata") or {}).get("doc_id")
        ],
        matched_chunk_ids=[item.get("chunk_id") for item in reranked_hits if item.get("chunk_id")],
        conflict_detected=bool(state.get("conflict_detected")),
        risk_level_hint=state.get("risk_level"),
    )


def decision_to_state_update(decision: RiskDecision) -> dict[str, Any]:
    """把风控决策收敛成可写入 RAGState 的字段。"""

    return {
        "risk_level": decision.risk_level,
        "risk_action": decision.action,
        "risk_reason": decision.reason,
        "risk_require_alert": decision.require_alert,
    }


class RuleBasedRiskEngine:
    """最小可落地的本地规则风控引擎。

    当前目标不是做完整企业风控平台，而是先把企业 RAG 最常见的安全动作做实：
    - 明显高风险批量导出直接 deny
    - restricted / local_only 强制本地
    - sensitive 最小化上下文
    - internal 可按策略脱敏
    """

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    def evaluate(self, context: RiskContext) -> RiskDecision:
        """根据请求阶段、数据分级和显式风险信号做决策。

        这里优先看三类信号：
        1. 明显高风险问法
        2. ACL / refusal 等上游已确定的安全语义
        3. data_classification / model_route
        """

        if not bool(getattr(self.settings, "enable_risk_engine", True)):
            return RiskDecision(
                allow=True,
                action="allow",
                risk_level=_merge_risk_level(context.risk_level_hint, assess_query_risk(context.question)),
                reason="risk_engine_disabled",
            )

        default_classification = getattr(self.settings, "default_data_classification", "internal")
        classification = normalize_classification(context.data_classification, default_classification)
        model_route = str(context.model_route or "").strip().lower()
        refusal_reason = str(context.refusal_reason or "").strip().lower()
        risk_level = _merge_risk_level(context.risk_level_hint, assess_query_risk(context.question))

        if self._looks_like_bulk_export(context.question):
            return RiskDecision(
                allow=False,
                action="deny",
                risk_level="high",
                reason="bulk_sensitive_export_request",
                require_alert=True,
                require_human_review=True,
            )

        if refusal_reason == "access_denied":
            return RiskDecision(
                allow=False,
                action="deny",
                risk_level=_merge_risk_level(risk_level, "high"),
                reason="access_denied",
                require_alert=True,
            )

        if classification == "restricted" or model_route == "local_only":
            return RiskDecision(
                allow=True,
                action="local_only",
                risk_level=_merge_risk_level(risk_level, "high"),
                reason="restricted_data_local_only",
                force_local_only=True,
                require_alert=bool(getattr(self.settings, "alert_on_restricted_access", True)),
            )

        if classification == "sensitive":
            return RiskDecision(
                allow=True,
                action="minimize",
                risk_level=_merge_risk_level(risk_level, "high"),
                reason="sensitive_context_minimized",
                require_redaction=True,
                max_context_chunks=int(getattr(self.settings, "sensitive_context_max_chunks", 1)),
                max_context_chars=int(getattr(self.settings, "sensitive_context_max_chars", 600)),
                require_alert=risk_level == "high",
            )

        if classification == "internal" and bool(
            getattr(self.settings, "internal_redact_for_external", True)
        ):
            return RiskDecision(
                allow=True,
                action="redact",
                risk_level=risk_level,
                reason="internal_context_redacted",
                require_redaction=True,
                require_alert=risk_level == "high",
            )

        return RiskDecision(
            allow=True,
            action="allow",
            risk_level=risk_level,
            reason="allow",
            require_alert=bool(context.conflict_detected) and bool(
                getattr(self.settings, "alert_on_conflict_detected", True)
            ),
        )

    def _looks_like_bulk_export(self, question: str) -> bool:
        """识别明显的高风险批量导出诉求。

        这类请求在企业场景里通常不应该继续交给模型自由处理，
        应尽量在入口或生成前直接打断。
        """

        text = question.strip()
        if not text:
            return False
        return any(pattern in text for pattern in _BULK_EXPORT_PATTERNS)
