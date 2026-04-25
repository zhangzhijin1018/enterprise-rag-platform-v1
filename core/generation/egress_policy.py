"""数据分级出域控制模块。

这层负责把“数据分级”和“模型路由”真正翻译成出域动作：
- 允许完整上下文
- 允许脱敏上下文
- 只允许最小必要片段
- 直接阻断外部模型

它是“企业安全策略”落到生成前的关键一层：
不是检索到了就能发给外部模型，而是要先经过这里判断。
"""

from __future__ import annotations

import re
from typing import Any

from core.retrieval.schemas import RetrievedChunk

_EMAIL_RE = re.compile(r"\b[\w.\-+]+@[\w.\-]+\.\w+\b")
_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_ID_CARD_RE = re.compile(r"(?<!\d)(\d{17}[\dXx]|\d{15})(?!\d)")
_BANK_CARD_RE = re.compile(r"(?<!\d)\d{16,19}(?!\d)")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def redact_text_for_external(text: str) -> str:
    """对准备出域的文本做最小必要脱敏。

    当前优先覆盖最常见的显式敏感字段：
    - 邮箱
    - 手机号
    - 身份证
    - 银行卡
    - IP

    这不是完整脱敏系统，而是生成前的“最小必要兜底脱敏”。
    """

    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    redacted = _PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    redacted = _ID_CARD_RE.sub("[REDACTED_ID]", redacted)
    redacted = _BANK_CARD_RE.sub("[REDACTED_ACCOUNT]", redacted)
    redacted = _IP_RE.sub("[REDACTED_IP]", redacted)
    return redacted


def _truncate_text(text: str, max_chars: int | None) -> str:
    """按字符数做保守截断，控制最小必要出域长度。

    对 sensitive 数据，不仅要脱敏，还要控制送出域的文本长度，
    避免把整段原文都暴露给外部模型。
    """

    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def prepare_contexts_for_generation(
    contexts: list[RetrievedChunk],
    *,
    settings: Any,
    data_classification: str | None,
    model_route: str | None,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """按数据分级和模型路由，收敛真正允许发给模型的上下文。

    当前策略大致是：
    - public / default: 完整上下文
    - internal: 可按配置脱敏
    - sensitive: 脱敏 + 限制 chunk 数 + 限制总长度
    - restricted / local_only: 不允许外部模型使用原始上下文

    你可以把返回值理解成两部分：
    - `prepared contexts`：真正允许送去生成的内容
    - `decision`：这次出域动作的解释和审计依据
    """

    classification = (data_classification or settings.default_data_classification or "internal").strip().lower()
    route = (model_route or "").strip().lower()
    local_only = {item.strip().lower() for item in getattr(settings, "local_only_classifications", [])}
    # 一旦 classification 或 route 明确要求 local_only，这里直接阻断外部出域。
    if classification in local_only or route == "local_only":
        # 一旦策略明确要求 local_only，就不要再尝试做“聪明降级出域”，
        # 直接把外部模型路径阻断，交给上层决定拒答还是本地生成。
        return [], {
            "allowed": False,
            "strategy": "local_only_blocked",
            "redacted": False,
            "truncated": False,
            "refusal_reason": "restricted_data_local_only",
        }

    max_chunks = len(contexts)
    max_chars: int | None = None
    should_redact = False
    strategy = "full_context"
    if classification == "internal":
        # internal 默认允许外部模型，但可以按配置先做脱敏。
        should_redact = bool(getattr(settings, "internal_redact_for_external", True))
        strategy = "redacted_context" if should_redact else "full_context"
    elif classification == "sensitive":
        # sensitive 默认只给最小必要上下文，避免把完整原文送出去。
        should_redact = True
        max_chunks = min(len(contexts), int(getattr(settings, "sensitive_context_max_chunks", 1)))
        max_chars = int(getattr(settings, "sensitive_context_max_chars", 600))
        strategy = "minimal_redacted_context"

    prepared: list[RetrievedChunk] = []
    for hit in contexts[:max_chunks]:
        content = hit.content
        if should_redact:
            content = redact_text_for_external(content)
        content = _truncate_text(content, max_chars)
        # trace 会把这次出域动作写回命中片段，方便后续 explainability / 审计复盘。
        trace = dict(hit.trace)
        trace["egress_strategy"] = strategy
        trace["egress_redacted"] = should_redact
        trace["egress_truncated"] = max_chars is not None
        prepared.append(hit.model_copy(update={"content": content, "trace": trace}))

    return prepared, {
        "allowed": True,
        "strategy": strategy,
        "redacted": should_redact,
        "truncated": max_chars is not None,
        "refusal_reason": "",
    }
