"""多轮上下文补全节点。

职责：
- 识别当前问题是否依赖历史上下文
- 在可补全时生成更完整的 `resolved_query`
- 无会话历史时安全退化，不阻断主链路
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.generation.prompts.templates import RESOLVE_CONTEXT_SYSTEM
from core.orchestration.state import RAGState
from core.services.runtime import RAGRuntime


_FOLLOWUP_RE = re.compile(r"^(那|那么|如果是|那如果|这个|那个|它|这条|这项|这类|这种|再|然后)")
_TIME_RE = re.compile(
    r"(今天|明天|后天|昨天|本周|下周|本月|上月|本季度|今年|去年|\d{1,2}月\d{1,2}[日号]?|\d{1,2}[日号])"
)
_SHIFT_RE = re.compile(r"(白班|夜班|早班|中班|晚班)")
_DEPARTMENT_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9_-]{2,30}(车间|部门|班组|小组|中心|事业部|工段|科室|仓库|产线|号线|线))"
)
_ENV_RE = re.compile(r"(生产|测试|预发|本地|线上|线下|docker|k8s|kubernetes|macos|linux|windows)", re.IGNORECASE)
_VERSION_RE = re.compile(r"(\b\d+\.\d+(\.\d+)?\b|版本|version)", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \n\t,.;，。；：")


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _extract_history_texts(state: RAGState) -> list[str]:
    """兼容多种上游 history 结构，但不强制变更当前 API。"""

    raw_history = (
        state.get("history_messages")
        or state.get("conversation_history")
        or state.get("chat_history")
        or state.get("messages")
        or []
    )
    if not isinstance(raw_history, list):
        return []

    out: list[str] = []
    for item in raw_history[-6:]:
        if isinstance(item, str):
            text = _normalize_text(item)
            if text:
                out.append(text)
            continue
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, str):
                text = _normalize_text(content)
                if text:
                    out.append(text)
    return out[-4:]


def _extract_context_slots(text: str) -> list[str]:
    """从最近一轮上下文提取适合补全到当前问题的关键约束。"""

    slots: list[str] = []
    for pattern in (_DEPARTMENT_RE, _SHIFT_RE, _TIME_RE, _ENV_RE, _VERSION_RE):
        match = pattern.search(text)
        if match:
            value = _normalize_text(match.group(1) if match.groups() else match.group(0))
            if value and value not in slots:
                slots.append(value)
    return slots


def _heuristic_resolve(question: str, history_texts: list[str]) -> str:
    """规则优先的上下文补全。

    当前策略不试图做复杂对话理解，只做最常见的省略补全：
    - 历史里已有部门 / 班次 / 时间 / 环境 / 版本等锚点
    - 当前问题用“这个 / 那个 / 那如果 / 今天谁值班”这类省略表达
    """

    q = _normalize_text(question)
    if not q or not history_texts:
        return ""

    last = history_texts[-1]
    need_resolution = bool(
        _FOLLOWUP_RE.search(q)
        or (bool(_TIME_RE.search(q)) and not bool(_DEPARTMENT_RE.search(q)))
    )
    if not need_resolution:
        return ""

    carry_slots: list[str] = []
    anchor_text = last
    for text in reversed(history_texts):
        slots = _extract_context_slots(text)
        if slots:
            carry_slots = slots
            anchor_text = text
            break
    if not carry_slots:
        return _normalize_text(f"{anchor_text} {q}")

    prefix = " ".join(slot for slot in carry_slots if slot not in q)
    resolved = _normalize_text(f"{prefix} {q}")
    return resolved if resolved and resolved != q else ""


async def resolve_context_node(state: RAGState, runtime: RAGRuntime) -> RAGState:
    """根据会话历史补全当前问题。"""

    question = _normalize_text(state.get("question") or "")
    signals = state.get("strategy_signals") or {}
    history_texts = _extract_history_texts(state)

    if not question:
        return {"resolved_query": ""}
    if not signals.get("need_history_resolution"):
        return {"resolved_query": ""}
    if not history_texts:
        return {"resolved_query": ""}

    history_payload = " || ".join(history_texts)
    cache_key = json.dumps(
        {
            "question": question,
            "history": history_texts,
            "strategy_signals": signals,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    cached = runtime.cache.get_json("resolve_context", cache_key)
    if isinstance(cached, dict):
        resolved_query = _normalize_text(str(cached.get("resolved_query") or ""))
        return {"resolved_query": resolved_query}

    heuristic = _heuristic_resolve(question, history_texts)
    if not runtime.llm.enabled:
        runtime.cache.set_json("resolve_context", cache_key, {"resolved_query": heuristic})
        return {"resolved_query": heuristic}

    messages = [
        {"role": "system", "content": RESOLVE_CONTEXT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"current_question: {question}\n"
                f"recent_history: {history_payload}\n"
                f"strategy_signals: {json.dumps(signals, ensure_ascii=False, sort_keys=True)}\n"
                "Return JSON only."
            ),
        },
    ]
    raw, _ = await runtime.llm.complete(
        messages,
        task="query_understanding",
        temperature=0.0,
        max_tokens=256,
    )
    obj = _extract_json_object(raw) or {}
    resolved_query = _normalize_text(str(obj.get("resolved_query") or ""))

    if resolved_query == question:
        resolved_query = ""
    if heuristic and not resolved_query:
        resolved_query = heuristic

    runtime.cache.set_json("resolve_context", cache_key, {"resolved_query": resolved_query})
    return {"resolved_query": resolved_query}
