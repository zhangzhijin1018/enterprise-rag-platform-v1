"""澄清判定节点模块。

这一步是检索规划之前的前置闸门，不属于 query route。

当前目标不再是“识别是不是报错问题”，而是判断：
- 当前问题是否缺少关键槽位
- 缺的是哪类槽位
- 是否应该先向用户追问，再进入检索链路
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from core.generation.prompts.templates import CLARIFY_DECISION_SYSTEM
from core.orchestration.state import RAGState
from core.services.runtime import RAGRuntime


@dataclass(slots=True)
class ClarifyDecision:
    """澄清判定结果。"""

    need_clarify: bool = False
    missing_slots: list[str] = field(default_factory=list)
    clarify_question: str = ""
    clarify_reason: str = ""


_ERROR_WORD_RE = re.compile(r"(报错|错误|异常|失败|问题|不生效|无法|不工作|故障)")
_ERROR_DETAIL_RE = re.compile(
    r"(\b(ERR|ERROR|E)[-_]?\d+\b|traceback|exception|stack trace|[A-Za-z]+Error|[A-Za-z]+Exception)",
    re.IGNORECASE,
)
_DEICTIC_RE = re.compile(r"^(这个|那个|它|这条|这项|这类|这种|那|那么|那如果|如果是)")
_COMPARE_RE = re.compile(r"(和|与|vs\.?|VS\.?|对比|比较|区别|差异)")
_VERSION_RE = re.compile(r"(\b\d+\.\d+(\.\d+)?\b|版本|version)", re.IGNORECASE)
_ENV_RE = re.compile(r"(生产|测试|预发|本地|线上|线下|docker|k8s|kubernetes|macos|linux|windows)", re.IGNORECASE)
_TIME_RE = re.compile(
    r"(今天|明天|后天|昨天|本周|下周|本月|上月|本季度|今年|去年|\d{1,2}月\d{1,2}[日号]?|\d{1,2}[日号])"
)
_SHIFT_RE = re.compile(r"(白班|夜班|早班|中班|晚班)")
_DEPARTMENT_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9_-]{2,30}(车间|部门|班组|小组|中心|事业部|工段|科室|仓库|产线|号线|线))"
)
_PERSON_RE = re.compile(r"([\u4e00-\u9fff]{2,4})(老师|同事|主管|经理|负责人)")
_SYMPTOM_RE = re.compile(r"(卡住|超时|失败|报错|无法|异常|中断|不生效|没结果|没返回)")


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip()


def _stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _has_history_context(state: RAGState) -> bool:
    for key in ("history_messages", "conversation_history", "chat_history", "messages"):
        raw = state.get(key)
        if isinstance(raw, list) and raw:
            return True
    return False


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


def _coerce_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
    return out


def _build_clarify_question(missing_slots: list[str]) -> str:
    slot_labels = {
        "target_object": "具体对象、模块或制度名称",
        "department": "部门/车间/产线",
        "time_range": "时间范围",
        "person": "人员姓名或角色",
        "environment": "运行环境",
        "version": "版本信息",
        "comparison_targets": "要比较的对象",
        "symptom_description": "现象描述",
        "runtime_context": "发生场景或上下文",
        "shift": "班次",
    }
    parts = [slot_labels.get(slot, slot) for slot in missing_slots]
    if not parts:
        return "请补充更具体的对象、背景或限制条件，我再继续帮你检索。"
    return f"请补充{ '、'.join(parts) }，这样我才能更准确地检索和回答。"


def _heuristic_clarify(
    question: str,
    strategy_signals: dict[str, Any],
    *,
    has_history_context: bool,
) -> ClarifyDecision:
    """规则版澄清判定，优先拦截缺槽位风险最高的问题。"""

    q = _normalize_question(question)
    missing_slots: list[str] = []

    if len(q) <= 4:
        missing_slots.append("target_object")

    if _DEICTIC_RE.search(q):
        if strategy_signals.get("need_history_resolution") and not has_history_context:
            missing_slots.append("runtime_context")
        if len(q) <= 12 and not has_history_context:
            missing_slots.append("target_object")

    if _COMPARE_RE.search(q):
        segments = re.split(r"(和|与|vs\.?|VS\.?|对比|比较|区别|差异)", q)
        if len([seg for seg in segments if seg.strip() and seg.strip() not in {"和", "与", "对比", "比较", "区别", "差异"}]) < 2:
            missing_slots.append("comparison_targets")

    if _ERROR_WORD_RE.search(q):
        if not _ERROR_DETAIL_RE.search(q):
            missing_slots.append("symptom_description")
        if not _ENV_RE.search(q):
            missing_slots.append("runtime_context")

    if re.search(r"(谁值班|谁上班|排班|班次|值班表|安排表)", q):
        if not _TIME_RE.search(q):
            missing_slots.append("time_range")
        if not _DEPARTMENT_RE.search(q) and not has_history_context:
            missing_slots.append("department")
        if "值班" in q and not _SHIFT_RE.search(q) and not has_history_context:
            missing_slots.append("shift")

    if re.search(r"(负责人|联系人|电话|归属谁|谁负责)", q):
        if not _DEPARTMENT_RE.search(q) and not _PERSON_RE.search(q):
            missing_slots.append("target_object")

    if re.search(r"(部署|安装|升级|兼容|接入)", q):
        if not _VERSION_RE.search(q):
            missing_slots.append("version")
        if not _ENV_RE.search(q):
            missing_slots.append("environment")

    if re.search(r"(怎么处理|怎么办|如何处理|如何解决)", q) and not _SYMPTOM_RE.search(q):
        missing_slots.append("symptom_description")

    missing_slots = list(dict.fromkeys(missing_slots))
    if not missing_slots:
        return ClarifyDecision()

    return ClarifyDecision(
        need_clarify=True,
        missing_slots=missing_slots,
        clarify_question=_build_clarify_question(missing_slots),
        clarify_reason=f"当前问题缺少关键槽位：{', '.join(missing_slots)}，直接检索容易误召回。",
    )


async def clarify_query_node(state: RAGState, runtime: RAGRuntime) -> RAGState:
    """判断当前问题是否应该先澄清再检索。"""

    question = _normalize_question(state.get("question") or "")
    strategy_signals = state.get("strategy_signals") or {}
    has_history_context = _has_history_context(state)
    cache_key = _stable_json(
        {
            "question": question,
            "strategy_signals": strategy_signals,
            "has_history_context": has_history_context,
        }
    )
    cached = runtime.cache.get_json("clarify", cache_key)
    if isinstance(cached, dict):
        return {
            "need_clarify": bool(cached.get("need_clarify")),
            "missing_slots": _coerce_list(cached.get("missing_slots")),
            "clarify_question": str(cached.get("clarify_question") or ""),
            "clarify_reason": str(cached.get("clarify_reason") or ""),
        }

    heuristic = _heuristic_clarify(
        question,
        strategy_signals,
        has_history_context=has_history_context,
    )
    if not runtime.llm.enabled:
        result = {
            "need_clarify": heuristic.need_clarify,
            "missing_slots": heuristic.missing_slots,
            "clarify_question": heuristic.clarify_question,
            "clarify_reason": heuristic.clarify_reason,
        }
        runtime.cache.set_json("clarify", cache_key, result)
        return result

    messages = [
        {"role": "system", "content": CLARIFY_DECISION_SYSTEM},
        {
            "role": "user",
            "content": (
                f"question: {question}\n"
                f"strategy_signals: {_stable_json(strategy_signals)}\n"
                f"has_history_context: {json.dumps(has_history_context, ensure_ascii=False)}\n"
                "Return JSON only."
            ),
        },
    ]
    raw, _ = await runtime.llm.complete(
        messages,
        task="query_understanding",
        temperature=0.0,
        max_tokens=320,
    )
    obj = _extract_json_object(raw) or {}

    need_clarify = bool(obj.get("need_clarify"))
    clarify_question = str(obj.get("clarify_question") or "").strip()
    clarify_reason = str(obj.get("clarify_reason") or "").strip()
    missing_slots = _coerce_list(obj.get("missing_slots"))

    if need_clarify and not clarify_question:
        need_clarify = heuristic.need_clarify
        clarify_question = heuristic.clarify_question
        clarify_reason = heuristic.clarify_reason
        missing_slots = heuristic.missing_slots

    if heuristic.need_clarify and not need_clarify:
        need_clarify = True
        clarify_question = heuristic.clarify_question
        clarify_reason = heuristic.clarify_reason
        missing_slots = heuristic.missing_slots

    result = {
        "need_clarify": need_clarify,
        "missing_slots": missing_slots,
        "clarify_question": clarify_question,
        "clarify_reason": clarify_reason,
    }
    runtime.cache.set_json("clarify", cache_key, result)
    return result


async def request_clarification_node(state: RAGState) -> RAGState:
    """把澄清判定结果转成最终可返回的响应状态。"""

    clarify_question = (
        state.get("clarify_question")
        or "当前问题信息不足，请补充更具体的对象、上下文或约束条件后再继续。"
    )
    clarify_reason = state.get("clarify_reason") or "问题缺少足够的检索锚点。"
    return {
        "answer": clarify_question,
        "confidence": 0.0,
        "reasoning_summary": clarify_reason,
        "citations": [],
        "reranked_hits": [],
        "refusal": True,
        "refusal_reason": "need_clarify",
        "grounding_ok": False,
    }
