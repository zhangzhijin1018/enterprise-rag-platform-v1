"""审计事件与高风险日志模块。

核心目标：
1. 对请求、prompt、output、response 形成统一审计事件
2. 日志中尽量避免落敏感明文
3. 对高风险 / 高敏 / 冲突 / 权限拒答补充安全告警事件
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from core.observability.logging import get_logger, get_request_log_context

from core.generation.egress_policy import redact_text_for_external

_HIGH_RISK_KEYWORDS = (
    "预算",
    "裁员",
    "薪酬",
    "工资",
    "招采",
    "采购合同",
    "收购",
    "名单",
    "身份证",
    "手机号",
    "密码",
    "导出全部",
    "完整名单",
    "编制调整",
)


def _hash_text(text: str) -> str:
    """对原始文本做不可逆摘要，用于审计留痕。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assess_query_risk(question: str) -> str:
    """根据问题内容给出粗粒度风险等级。"""

    q = question.strip()
    if not q:
        return "low"
    if any(keyword in q for keyword in _HIGH_RISK_KEYWORDS):
        return "high"
    if any(token in q for token in ("审批", "人员", "报销", "预算", "制度", "采购")):
        return "medium"
    return "low"


def _preview_text(text: str, *, preview_chars: int, redact: bool) -> str:
    """生成审计预览文本。

    默认会先做脱敏，再做长度裁剪，避免日志里直接落整段原文。
    """

    content = redact_text_for_external(text) if redact else text
    if len(content) <= preview_chars:
        return content
    return content[: preview_chars - 1].rstrip() + "…"


def build_audit_event(
    *,
    stage: str,
    audit_id: str,
    question: str,
    user_context: dict[str, Any] | None,
    settings: Any,
    state: dict[str, Any] | None = None,
    prompt: str | None = None,
    output: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造统一审计事件。

    这一步统一收敛：
    - 请求身份
    - 风控状态
    - 检索命中
    - prompt/output 摘要
    - 拒答 / 冲突 / 模型路由
    方便后续做追责、排查和复盘。
    """

    state = state or {}
    user_context = user_context or {}
    preview_chars = int(getattr(settings, "audit_log_preview_chars", 240))
    redact_content = bool(getattr(settings, "audit_log_redact_content", True))
    reranked_hits = state.get("reranked_hits") or []
    trace_id = str(
        state.get("trace_id")
        or get_request_log_context().get("trace_id")
        or ""
    ).strip()
    return {
        "event_type": stage,
        "trace_id": trace_id or None,
        "audit_id": audit_id,
        "risk_level": state.get("risk_level") or assess_query_risk(question),
        "risk_action": state.get("risk_action"),
        "risk_reason": state.get("risk_reason"),
        "risk_require_alert": bool(state.get("risk_require_alert")),
        "question_hash": _hash_text(question or ""),
        "question_preview": _preview_text(
            question or "",
            preview_chars=preview_chars,
            redact=redact_content,
        ),
        "user_id": user_context.get("user_id"),
        "department": user_context.get("department"),
        "role": user_context.get("role"),
        "project_ids": user_context.get("project_ids") or [],
        "data_classification": state.get("data_classification"),
        "model_route": state.get("model_route"),
        "refusal": bool(state.get("refusal")),
        "refusal_reason": state.get("refusal_reason"),
        "conflict_detected": bool(state.get("conflict_detected")),
        "conflict_summary": state.get("conflict_summary"),
        "retrieved_chunk_ids": [item.get("chunk_id") for item in reranked_hits if item.get("chunk_id")],
        "retrieved_doc_ids": [
            (item.get("metadata") or {}).get("doc_id")
            for item in reranked_hits
            if (item.get("metadata") or {}).get("doc_id")
        ],
        "prompt_hash": _hash_text(prompt) if prompt else None,
        "prompt_preview": (
            _preview_text(prompt, preview_chars=preview_chars, redact=redact_content)
            if prompt
            else None
        ),
        "output_hash": _hash_text(output) if output else None,
        "output_preview": (
            _preview_text(output, preview_chars=preview_chars, redact=redact_content)
            if output
            else None
        ),
        **(extra or {}),
    }


def log_audit_event(
    *,
    stage: str,
    audit_id: str,
    question: str,
    user_context: dict[str, Any] | None,
    settings: Any,
    state: dict[str, Any] | None = None,
    prompt: str | None = None,
    output: str | None = None,
    extra: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """记录统一审计日志。

    当前是结构化 JSON 日志输出；
    后续如果接外部审计平台，这里仍然可以作为统一事件出口。
    """

    event = build_audit_event(
        stage=stage,
        audit_id=audit_id,
        question=question,
        user_context=user_context,
        settings=settings,
        state=state,
        prompt=prompt,
        output=output,
        extra=extra,
    )
    if not getattr(settings, "audit_log_enabled", True):
        return event
    audit_logger = logger or get_logger("audit")
    level = logging.WARNING if event["risk_level"] == "high" else logging.INFO
    audit_logger.log(level, json.dumps(event, ensure_ascii=False), extra={"event": stage})
    return event


def should_trigger_alert(
    *,
    settings: Any,
    state: dict[str, Any] | None = None,
    risk_level: str | None = None,
) -> bool:
    """判断当前请求是否需要额外安全告警。

    告警和普通审计事件不同：
    - 普通审计强调留痕
    - 告警强调“需要额外关注”
    """

    state = state or {}
    level = (risk_level or "").strip().lower()
    classification = str(state.get("data_classification") or "").strip().lower()
    refusal_reason = str(state.get("refusal_reason") or "").strip().lower()

    if bool(state.get("risk_require_alert")):
        return True
    if level == "high" and getattr(settings, "alert_on_high_risk_queries", True):
        return True
    if classification == "restricted" and getattr(settings, "alert_on_restricted_access", True):
        return True
    if refusal_reason in {"access_denied", "restricted_data_local_only"} and getattr(
        settings, "alert_on_restricted_access", True
    ):
        return True
    if bool(state.get("conflict_detected")) and getattr(settings, "alert_on_conflict_detected", True):
        return True
    return False


def log_alert_event(
    *,
    audit_id: str,
    question: str,
    user_context: dict[str, Any] | None,
    settings: Any,
    state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any] | None:
    """按告警条件输出一条额外安全事件。

    当前仍是结构化日志分流；
    后续可以很自然地替换成 webhook / Kafka / SIEM 对接。
    """

    risk_level = assess_query_risk(question)
    if not should_trigger_alert(settings=settings, state=state, risk_level=risk_level):
        return None
    event = build_audit_event(
        stage="security_alert",
        audit_id=audit_id,
        question=question,
        user_context=user_context,
        settings=settings,
        state=state,
        extra={"risk_level": risk_level, **(extra or {})},
    )
    if getattr(settings, "audit_log_enabled", True):
        alert_logger = logger or get_logger("audit.alert")
    alert_logger.warning(json.dumps(event, ensure_ascii=False), extra={"event": "security_alert"})
    return event
