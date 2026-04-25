"""查询策略信号分析节点。

这个节点不再输出固定业务分类，而是把问题映射成更通用的策略信号：

- 是否疑似依赖历史上下文
- 是否适合增强关键词召回
- 是否需要子查询拆分
- 是否适合启用 HyDE
- 是否像时间/部门/人员等结构化事实查询

后续 `clarify / resolve_context / build_query_plan / retrieve` 都基于这些信号做决策。
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.generation.prompts.templates import QUERY_UNDERSTANDING_SYSTEM
from core.observability import get_logger
from core.orchestration.query_understanding_vocab import (
    QueryUnderstandingIndex,
    load_query_understanding_index,
    load_query_understanding_vocab,
)
from core.orchestration.state import RAGState
from core.services.runtime import RAGRuntime

logger = get_logger(__name__)


_ERROR_CODE_RE = re.compile(r"\b(ERR|ERROR|E)[-_]?\d+\b", re.IGNORECASE)
_VERSION_RE = re.compile(r"(\b\d+\.\d+(\.\d+)?\b|版本|version)", re.IGNORECASE)
_ENV_RE = re.compile(r"(生产|测试|预发|本地|线上|线下|docker|k8s|kubernetes|macos|linux|windows)", re.IGNORECASE)
_TIME_RE = re.compile(
    r"(今天|明天|后天|昨天|本周|下周|本月|上月|本季度|今年|去年|\d{1,2}月\d{1,2}[日号]?|\d{1,2}[日号])"
)
_PERSON_RE = re.compile(r"([\u4e00-\u9fff]{2,4})(老师|同事|主管|经理|负责人)")
_DOC_NUMBER_RE = re.compile(r"[A-Za-z]{1,10}[-_/][A-Za-z0-9\u4e00-\u9fff-_/]{2,32}")
_PRECISE_IDENTIFIER_RE = re.compile(
    r"\b[A-Z][A-Z0-9._/-]{2,}\b|\b\d{3,}\b|`[^`]{2,40}`|“[^”]{2,40}”|\"[^\"]{2,40}\""
)
# 这一组 regex 主要负责识别“强锚点”：
# - 错误码
# - 版本号
# - 文号
# - 时间约束
# - 精确标识
# 它们会直接影响 query_scene、preferred_retriever、top_k_profile。


def _normalize_question(question: str) -> str:
    """对问题做轻量规整，降低后续规则匹配噪声。"""

    return re.sub(r"\s+", " ", question).strip()


def _infer_business_domain_from_index(question: str, index: QueryUnderstandingIndex) -> str | None:
    """从预编译 business domain 索引里推断最可能的业务域。"""

    best: tuple[int, int, int, str] | None = None
    for domain, keywords in index.business_domain_entries:
        hits: list[tuple[int, int]] = []
        for keyword in keywords:
            pos = question.find(keyword)
            if pos >= 0:
                hits.append((pos, len(keyword)))
        if not hits:
            continue
        hit_count = len(hits)
        earliest_pos = min(item[0] for item in hits)
        longest_keyword = max(item[1] for item in hits)
        candidate = (hit_count, -earliest_pos, longest_keyword, str(domain).strip())
        if best is None or candidate > best:
            best = candidate
    return best[3] if best else None


def _extract_alias_match(
    question: str,
    entries: tuple[tuple[str, tuple[str, ...]], ...] | None,
) -> str | None:
    """从别名字典里抽取最合适的规范值。

    当前匹配策略更偏“最长 alias + 最早命中 + 命中数更多”。
    这样能减少短词、泛词带来的误判。
    """

    if not entries:
        return None
    best: tuple[int, int, int, str] | None = None
    for canonical, normalized_aliases in entries:
        hits: list[tuple[int, int]] = []
        for alias in normalized_aliases:
            pos = question.find(alias)
            if pos >= 0:
                hits.append((pos, len(alias)))
        if not hits:
            continue
        hit_count = len(hits)
        earliest_pos = min(item[0] for item in hits)
        longest_alias = max(item[1] for item in hits)
        candidate = (hit_count, -earliest_pos, longest_alias, str(canonical).strip())
        if best is None or candidate > best:
            best = candidate
    return best[3] if best else None


def _extract_strategy_signals(
    question: str,
    vocab: dict[str, Any] | None = None,
    index: QueryUnderstandingIndex | None = None,
) -> dict[str, Any]:
    """基于轻量规则抽取策略信号。

    这一步不是最终语义理解，而是第一层检索规划信号提取。
    输出的信号主要服务于：
    - clarify
    - resolve_context
    - build_query_plan
    - retrieve
    """

    index = index or load_query_understanding_index()
    vocab = vocab or index.vocab
    q = _normalize_question(question)
    patterns = index.scene_patterns
    department_re = index.department_pattern
    equipment_re = index.equipment_pattern
    department_alias = _extract_alias_match(q, index.department_alias_entries)
    site_alias = _extract_alias_match(q, index.site_alias_entries)
    system_alias = _extract_alias_match(q, index.system_alias_entries)
    has_error_code = bool(_ERROR_CODE_RE.search(q))
    has_time_constraint = bool(_TIME_RE.search(q))
    has_department_constraint = bool(department_re.search(q) or department_alias)
    has_person_constraint = bool(_PERSON_RE.search(q))
    has_environment_constraint = bool(_ENV_RE.search(q))
    has_version_constraint = bool(_VERSION_RE.search(q))
    has_shift_constraint = bool(patterns["shift"].search(q))
    has_doc_number = bool(_DOC_NUMBER_RE.search(q))
    has_equipment_constraint = bool(equipment_re.search(q))
    is_comparison = bool(patterns["compare"].search(q))
    has_precise_identifier = bool(
        has_error_code
        or has_version_constraint
        or has_doc_number
        or _PRECISE_IDENTIFIER_RE.search(q)
        or department_re.search(q)
    )
    likely_structured_lookup = bool(
        patterns["structured_fact"].search(q)
        or (
            has_time_constraint
            and (has_department_constraint or has_person_constraint or has_shift_constraint)
        )
    )
    need_history_resolution = bool(
        patterns["followup"].search(q)
        or (
            has_time_constraint
            and not has_department_constraint
            and bool(re.search(r"(谁|值班|排班|安排|负责人|电话|在哪|哪里)", q))
        )
    )
    is_multi_hop = bool(
        is_comparison
        or re.search(r"(以及|并且|同时|分别|先.*再|不仅.*还|原因.*处理)", q)
    )
    need_sub_queries = bool(is_multi_hop or patterns["sub_query"].search(q))
    need_hyde = bool(
        patterns["hyde"].search(q) and not likely_structured_lookup and not has_precise_identifier
    )
    need_keyword_boost = bool(
        has_precise_identifier
        or likely_structured_lookup
        or re.search(r"[A-Za-z0-9._/-]{3,}", q)
    )
    need_clarify = bool(len(q) <= 4 or re.fullmatch(r"(这个|那个|它|这个问题|这个报错)", q))

    metadata_intent: dict[str, Any] = {}
    # `metadata_intent` 表示“这题更像要检什么”：
    # - 可能是 doc_number
    # - 可能是 department / plant / system_name
    # - 也可能是 business_domain / equipment_type
    # 它既会影响 retrieval filter，也会影响后续 boost。
    query_scene = "general_lookup"
    preferred_retriever = "hybrid"
    top_k_profile = "balanced"

    if patterns["policy"].search(q):
        query_scene = "policy_lookup"
        preferred_retriever = "sparse"
        top_k_profile = "precise"
        metadata_intent["doc_category"] = "policy"
    elif patterns["procedure"].search(q):
        query_scene = "procedure_lookup"
        preferred_retriever = "hybrid"
        top_k_profile = "balanced"
        metadata_intent["doc_category"] = "procedure"
    elif patterns["meeting"].search(q):
        query_scene = "meeting_trace"
        preferred_retriever = "dense"
        top_k_profile = "broad"
        metadata_intent["doc_category"] = "meeting"
    elif patterns["project"].search(q):
        query_scene = "project_trace"
        preferred_retriever = "hybrid"
        top_k_profile = "broad"
        metadata_intent["business_domain"] = "project_management"
    elif has_error_code:
        query_scene = "error_code_lookup"
        preferred_retriever = "sparse"
        top_k_profile = "precise"
    elif likely_structured_lookup:
        query_scene = "structured_fact_lookup"
        preferred_retriever = "sparse"
        top_k_profile = "precise"
    elif is_comparison or is_multi_hop:
        query_scene = "comparison_analysis"
        preferred_retriever = "hybrid"
        top_k_profile = "broad"

    inferred_domain = _infer_business_domain_from_index(q, index)
    if department_alias:
        metadata_intent["department"] = department_alias
        metadata_intent["owner_department"] = department_alias
    if site_alias:
        metadata_intent["plant"] = site_alias
        metadata_intent["applicable_site"] = site_alias
    if system_alias:
        metadata_intent["system_name"] = system_alias

    if has_equipment_constraint:
        metadata_intent["business_domain"] = (
            metadata_intent.get("business_domain")
            or inferred_domain
            or "equipment_maintenance"
        )
        if patterns["procedure"].search(q):
            metadata_intent["process_stage"] = "inspection"
        equipment_match = equipment_re.search(q)
        if equipment_match:
            metadata_intent["equipment_type"] = equipment_match.group(1)
    if inferred_domain:
        if metadata_intent.get("business_domain") in {None, "", "equipment_maintenance"}:
            metadata_intent["business_domain"] = inferred_domain
    if patterns["system"].search(q):
        metadata_intent["business_domain"] = metadata_intent.get("business_domain") or "it_ops"
    if has_doc_number:
        metadata_intent["doc_number"] = _DOC_NUMBER_RE.search(q).group(0)

    return {
        "need_clarify": need_clarify,
        "need_history_resolution": need_history_resolution,
        "need_keyword_boost": need_keyword_boost,
        "need_sub_queries": need_sub_queries,
        "need_hyde": need_hyde,
        "likely_structured_lookup": likely_structured_lookup,
        "has_precise_identifier": has_precise_identifier,
        "has_time_constraint": has_time_constraint,
        "has_department_constraint": has_department_constraint,
        "has_person_constraint": has_person_constraint,
        "has_environment_constraint": has_environment_constraint,
        "has_version_constraint": has_version_constraint,
        "has_shift_constraint": has_shift_constraint,
        "is_comparison": is_comparison,
        "is_multi_hop": is_multi_hop,
        "has_error_code": has_error_code,
        "has_doc_number": has_doc_number,
        "has_equipment_constraint": has_equipment_constraint,
        "query_scene": query_scene,
        "preferred_retriever": preferred_retriever,
        "top_k_profile": top_k_profile,
        "metadata_intent": metadata_intent,
    }


def _clamp_confidence(value: object, *, default: float) -> float:
    """把任意输入收敛到 0-1 区间。"""

    try:
        conf = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, conf))


def _heuristic_confidence(question: str, signals: dict[str, Any]) -> tuple[float, str]:
    """评估规则判断的可靠度。

    这是 query understanding 的第一层置信度估计：
    - 高置信：直接用规则结果
    - 中低置信：允许调 LLM 补判
    - 很低置信：后面会触发 guardrail，回退到保守策略
    """

    q = _normalize_question(question)
    score = 0.28
    reasons: list[str] = []

    if signals.get("has_doc_number"):
        score += 0.26
        reasons.append("命中文号")
    if signals.get("has_error_code"):
        score += 0.24
        reasons.append("命中错误码")
    if signals.get("likely_structured_lookup"):
        score += 0.18
        reasons.append("像结构化事实查询")
    if signals.get("has_precise_identifier"):
        score += 0.12
        reasons.append("包含精确标识")
    if signals.get("has_equipment_constraint"):
        score += 0.08
        reasons.append("命中设备锚点")
    if signals.get("query_scene") in {
        "policy_lookup",
        "procedure_lookup",
        "meeting_trace",
        "error_code_lookup",
        "structured_fact_lookup",
    }:
        score += 0.18
        reasons.append("命中强模式查询场景")
    if (signals.get("metadata_intent") or {}).get("doc_category") in {"policy", "procedure", "meeting"}:
        score += 0.12
        reasons.append("文档类别锚点明确")
    if signals.get("is_comparison"):
        score += 0.08
        reasons.append("命中对比信号")
    if signals.get("has_time_constraint") and signals.get("has_department_constraint"):
        score += 0.08
        reasons.append("时间和组织约束完整")

    if len(q) >= 32:
        score -= 0.06
    if signals.get("is_multi_hop"):
        score -= 0.08
    if len(signals.get("metadata_intent") or {}) >= 3:
        score -= 0.05
    if (
        signals.get("query_scene") in {"general_lookup", "project_trace"}
        and not signals.get("has_precise_identifier")
        and not signals.get("likely_structured_lookup")
    ):
        score -= 0.08

    confidence = max(0.18, min(0.96, score))
    reason = "；".join(reasons) if reasons else "主要依赖通用启发式"
    return confidence, reason


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """尽量从文本里提取 JSON 对象。"""

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


def _coerce_signal_bool(value: object, fallback: bool) -> bool:
    """把 LLM 输出收敛成 bool signal。"""

    if isinstance(value, bool):
        return value
    return fallback


def _coerce_metadata_intent(value: object, fallback: dict[str, Any]) -> dict[str, Any]:
    """把 LLM 输出收敛成稳定的 metadata_intent。"""

    if not isinstance(value, dict):
        return dict(fallback)
    out: dict[str, Any] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(raw, (str, int, float, bool)):
            text = str(raw).strip()
            if text:
                out[key] = text
        elif isinstance(raw, list):
            vals = [str(item).strip() for item in raw if str(item).strip()]
            if vals:
                out[key] = vals
    merged = dict(fallback)
    merged.update(out)
    return merged


def _merge_llm_signals(heuristic: dict[str, Any], llm_obj: dict[str, Any]) -> dict[str, Any]:
    """用 LLM 修正规则判断，但保留显式锚点。

    设计原则：
    - 规则层负责快、稳、可解释
    - LLM 层负责补长尾表达和复杂指代
    - 不能让 LLM 直接抹掉明显的强锚点
    """

    merged = dict(heuristic)
    for key in (
        "need_history_resolution",
        "need_sub_queries",
        "need_hyde",
        "need_keyword_boost",
    ):
        merged[key] = _coerce_signal_bool(llm_obj.get(key), bool(heuristic.get(key)))

    query_scene = str(llm_obj.get("query_scene") or heuristic.get("query_scene") or "general_lookup")
    if query_scene in {
        "general_lookup",
        "policy_lookup",
        "procedure_lookup",
        "meeting_trace",
        "project_trace",
        "error_code_lookup",
        "structured_fact_lookup",
        "comparison_analysis",
    }:
        merged["query_scene"] = query_scene

    preferred = str(llm_obj.get("preferred_retriever") or heuristic.get("preferred_retriever") or "hybrid")
    if preferred in {"sparse", "dense", "hybrid"}:
        merged["preferred_retriever"] = preferred

    profile = str(llm_obj.get("top_k_profile") or heuristic.get("top_k_profile") or "balanced")
    if profile in {"precise", "balanced", "broad"}:
        merged["top_k_profile"] = profile

    merged["metadata_intent"] = _coerce_metadata_intent(
        llm_obj.get("metadata_intent"),
        fallback=dict(heuristic.get("metadata_intent") or {}),
    )
    return merged


def _apply_low_confidence_guardrail(signals: dict[str, Any]) -> dict[str, Any]:
    """很低置信度时回退到保守检索。

    当前 guardrail 的核心目标是：
    - 少误路由
    - 少过度启用 HyDE
    - 在判断不稳时尽量回到更安全的 `hybrid + balanced`
    """

    guarded = dict(signals)
    guarded["preferred_retriever"] = "hybrid"
    guarded["top_k_profile"] = "balanced"
    guarded["need_hyde"] = False
    if not guarded.get("has_precise_identifier") and not guarded.get("likely_structured_lookup"):
        guarded["query_scene"] = "general_lookup"
    return guarded


async def analyze_query_node(state: RAGState, runtime: RAGRuntime | None = None) -> RAGState:
    """输出面向检索规划的策略信号。

    执行顺序：
    1. 规则层抽取 `strategy_signals`
    2. 计算 heuristic confidence
    3. 命中缓存则直接返回
    4. 低置信时再调用 query understanding 模型
    5. 很低置信时应用 guardrail，回退到保守路由
    """

    question = state.get("question") or ""
    logger.info("query analysis started", extra={"event": "query_analysis_started"})
    if runtime is not None:
        index = load_query_understanding_index(runtime.settings)
        vocab = index.vocab
    else:
        vocab = load_query_understanding_vocab()
        index = load_query_understanding_index()
    signals = _extract_strategy_signals(question, vocab, index)
    confidence, reason = _heuristic_confidence(question, signals)

    if runtime is not None:
        cache_payload = json.dumps(
            {
                "question": question,
                "query_understanding_vocab_path": getattr(
                    runtime.settings,
                    "query_understanding_vocab_path",
                    "",
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cached = runtime.cache.get_json("query_understanding", cache_payload)
        if isinstance(cached, dict):
            cached_signals = dict(cached.get("strategy_signals") or {})
            result = {
                "strategy_signals": cached_signals,
                "metadata_intent": dict(cached_signals.get("metadata_intent") or {}),
                "analysis_confidence": _clamp_confidence(
                    cached.get("analysis_confidence"),
                    default=confidence,
                ),
                "analysis_source": str(cached.get("analysis_source") or "heuristic_cache"),
                "analysis_reason": str(cached.get("analysis_reason") or reason),
            }
            logger.info(
                "query analysis completed from cache",
                extra={
                    "event": "query_analysis_completed",
                    "analysis_source": result["analysis_source"],
                    "query_scene": cached_signals.get("query_scene"),
                    "preferred_retriever": cached_signals.get("preferred_retriever"),
                },
            )
            return result

        threshold = float(getattr(runtime.settings, "query_understanding_confidence_threshold", 0.78))
        guardrail_threshold = float(
            getattr(runtime.settings, "query_understanding_force_hybrid_threshold", 0.58)
        )
        source = "heuristic"

        if runtime.llm.enabled and confidence < threshold:
            messages = [
                {"role": "system", "content": QUERY_UNDERSTANDING_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"question: {question}\n"
                        f"heuristic_signals: {json.dumps(signals, ensure_ascii=False, sort_keys=True)}\n"
                        "Return JSON only."
                    ),
                },
            ]
            raw, _ = await runtime.llm.complete(
                messages,
                task="query_understanding",
                temperature=0.0,
                max_tokens=int(getattr(runtime.settings, "query_understanding_max_tokens", 512)),
            )
            llm_obj = _extract_json_object(raw) or {}
            if llm_obj:
                signals = _merge_llm_signals(signals, llm_obj)
                confidence = max(
                    confidence,
                    _clamp_confidence(llm_obj.get("confidence"), default=confidence),
                )
                reason = str(llm_obj.get("reason") or reason).strip() or reason
                source = "llm_enhanced"
            else:
                source = "heuristic_fallback"

        if confidence < guardrail_threshold:
            signals = _apply_low_confidence_guardrail(signals)
            source = f"{source}_guardrail"

        runtime.cache.set_json(
            "query_understanding",
            cache_payload,
            {
                "strategy_signals": signals,
                "analysis_confidence": confidence,
                "analysis_source": source,
                "analysis_reason": reason,
            },
        )
        result = {
            "strategy_signals": signals,
            "metadata_intent": dict(signals.get("metadata_intent") or {}),
            "analysis_confidence": confidence,
            "analysis_source": source,
            "analysis_reason": reason,
        }
        logger.info(
            "query analysis completed",
            extra={
                "event": "query_analysis_completed",
                "analysis_source": source,
                "query_scene": signals.get("query_scene"),
                "preferred_retriever": signals.get("preferred_retriever"),
            },
        )
        return result

    result = {
        "strategy_signals": signals,
        "metadata_intent": dict(signals.get("metadata_intent") or {}),
        "analysis_confidence": confidence,
        "analysis_source": "heuristic",
        "analysis_reason": reason,
    }
    logger.info(
        "query analysis completed",
        extra={
            "event": "query_analysis_completed",
            "analysis_source": "heuristic",
            "query_scene": signals.get("query_scene"),
            "preferred_retriever": signals.get("preferred_retriever"),
        },
    )
    return result
