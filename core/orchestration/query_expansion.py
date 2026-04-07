"""查询规划与多路查询扩展模块。

和第一版“只有一个 rewrite query”的实现相比，这个模块现在负责输出一份更完整的检索计划：

- `rewritten_query`：主检索 query
- `multi_queries`：子问题或替代表达
- `keyword_queries`：更适合 BM25 的短 query
- `hyde_query`：更适合 dense retrieval 的假设答案

设计目标不是生成越多 query 越好，而是让不同检索器拿到更适合自己的输入。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from core.generation.prompts.templates import QUERY_PLAN_SYSTEM
from core.services.runtime import RAGRuntime


@dataclass(slots=True)
class QueryPlan:
    """查询规划结果。"""

    rewritten_query: str
    multi_queries: list[str] = field(default_factory=list)
    keyword_queries: list[str] = field(default_factory=list)
    hyde_query: str = ""
    planning_summary: str = ""


_COMPARE_RE = re.compile(r"(.+?)(和|与|vs\.?|VS\.?|对比|比较)(.+)")
_ERROR_CODE_RE = re.compile(r"\b(ERR|ERROR|E)[-_]?\d+\b", re.IGNORECASE)
_QUOTED_TERM_RE = re.compile(r"[\"“”'`【】\[\]]([^\"“”'`【】\[\]]{2,40})[\"“”'`【】\[\]]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9._/-]{2,}|[\u4e00-\u9fff]{2,}")


def _normalize_query(text: str) -> str:
    """做轻量文本收敛，避免 query 带冗余空白和尾部噪声。"""

    return re.sub(r"\s+", " ", text).strip(" \n\t,.;，。；：")


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """按原顺序去重。"""

    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = _normalize_query(item)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _extract_json_object(raw: str) -> dict | None:
    """从 LLM 文本中尽量提取一个 JSON 对象。"""

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


def _coerce_str_list(value: object, *, limit: int) -> list[str]:
    """把模型输出尽量收敛成字符串列表，并限制数量。"""

    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = _normalize_query(item)
            if text:
                out.append(text)
        if len(out) >= limit:
            break
    return _dedupe_keep_order(out)


def _extract_keyword_candidates(question: str) -> list[str]:
    """提取适合稀疏检索的关键词候选。"""

    out = [m.group(0) for m in _ERROR_CODE_RE.finditer(question)]
    out.extend(m.group(1) for m in _QUOTED_TERM_RE.finditer(question))
    for token in _TOKEN_RE.findall(question):
        if re.search(r"[A-Za-z0-9._/-]", token):
            out.append(token)
    return _dedupe_keep_order(out)[:4]


def _heuristic_query_plan(question: str, *, query_type: str = "general") -> QueryPlan:
    """规则版查询规划。"""

    question = _normalize_query(question)
    rewritten = question
    multi_queries: list[str] = []
    keyword_queries = _extract_keyword_candidates(question)
    hyde_query = ""
    planning_summary = "保持原问题为主，仅做轻量多路查询扩展。"

    error_codes = [m.group(0) for m in _ERROR_CODE_RE.finditer(question)]
    if query_type == "error_code" and error_codes:
        code = error_codes[0]
        rewritten = f"{question} {code} 原因 排查 处理方法"
        multi_queries.extend([f"{code} 是什么", f"{code} 原因", f"{code} 处理方法"])
        keyword_queries.extend(error_codes)
        planning_summary = "识别为错误码问题，补充原因与处理方法子查询。"
    elif query_type == "procedure":
        rewritten = f"{question} SOP 步骤 前置条件 注意事项"
        multi_queries.extend([f"{question} 步骤", f"{question} 前置条件", f"{question} 注意事项"])
        keyword_queries.extend([question, f"{question} SOP"])
        planning_summary = "识别为流程问题，扩展步骤、前置条件和注意事项。"
    else:
        match = _COMPARE_RE.search(question)
        if match:
            left = _normalize_query(match.group(1))
            right = _normalize_query(match.group(3))
            rewritten = f"{left} {right} 区别 对比 适用场景"
            multi_queries.extend(
                [
                    f"{left} 特点 适用场景",
                    f"{right} 特点 适用场景",
                    f"{left} {right} 区别",
                ]
            )
            keyword_queries.extend([left, right, f"{left} {right}"])
            planning_summary = "识别为对比类问题，拆成对象特性与差异三路查询。"
        elif re.search(r"(为什么|原因|排查|如何处理|怎么处理|架构|部署)", question):
            multi_queries.extend([f"{question} 原因", f"{question} 处理方法"])
            planning_summary = "识别为原因或方案类问题，补充原因与处理方向子查询。"

    return QueryPlan(
        rewritten_query=rewritten,
        multi_queries=_dedupe_keep_order(multi_queries)[:3],
        keyword_queries=_dedupe_keep_order(keyword_queries)[:4],
        hyde_query=hyde_query,
        planning_summary=planning_summary,
    )


async def build_query_plan(
    runtime: RAGRuntime,
    *,
    question: str,
    query_type: str = "general",
) -> QueryPlan:
    """生成一份查询计划。

    执行顺序：
    1. 命中缓存则直接返回；
    2. 无 LLM 时走规则版规划；
    3. 有 LLM 时尝试结构化生成，并用规则版做字段级兜底。
    """

    question = _normalize_query(question)
    cache_key = f"{query_type}:{question}"
    cached = runtime.cache.get_json("query_plan", cache_key)
    if isinstance(cached, dict):
        return QueryPlan(
            rewritten_query=str(cached.get("rewritten_query") or question),
            multi_queries=_coerce_str_list(cached.get("multi_queries"), limit=3),
            keyword_queries=_coerce_str_list(cached.get("keyword_queries"), limit=4),
            hyde_query=_normalize_query(str(cached.get("hyde_query") or "")),
            planning_summary=str(cached.get("planning_summary") or ""),
        )

    heuristic = _heuristic_query_plan(question, query_type=query_type)
    if not runtime.llm.enabled:
        runtime.cache.set_json(
            "query_plan",
            cache_key,
            {
                "rewritten_query": heuristic.rewritten_query,
                "multi_queries": heuristic.multi_queries,
                "keyword_queries": heuristic.keyword_queries,
                "hyde_query": heuristic.hyde_query,
                "planning_summary": heuristic.planning_summary,
            },
        )
        return heuristic

    messages = [
        {"role": "system", "content": QUERY_PLAN_SYSTEM},
        {
            "role": "user",
            "content": f"question: {question}\nquery_type: {query_type}\nReturn JSON only.",
        },
    ]
    raw, _ = await runtime.llm.complete(messages, temperature=0.0, max_tokens=512)
    obj = _extract_json_object(raw) or {}

    rewritten_query = _normalize_query(str(obj.get("rewritten_query") or heuristic.rewritten_query))
    multi_queries = _coerce_str_list(obj.get("multi_queries"), limit=3) or heuristic.multi_queries
    keyword_queries = _coerce_str_list(obj.get("keyword_queries"), limit=4) or heuristic.keyword_queries
    hyde_query = _normalize_query(str(obj.get("hyde_query") or ""))
    planning_summary = _normalize_query(str(obj.get("planning_summary") or heuristic.planning_summary))

    if not rewritten_query:
        rewritten_query = heuristic.rewritten_query or question

    plan = QueryPlan(
        rewritten_query=rewritten_query,
        multi_queries=_dedupe_keep_order(multi_queries)[:3],
        keyword_queries=_dedupe_keep_order(keyword_queries)[:4],
        hyde_query=hyde_query,
        planning_summary=planning_summary,
    )
    runtime.cache.set_json(
        "query_plan",
        cache_key,
        {
            "rewritten_query": plan.rewritten_query,
            "multi_queries": plan.multi_queries,
            "keyword_queries": plan.keyword_queries,
            "hyde_query": plan.hyde_query,
            "planning_summary": plan.planning_summary,
        },
    )
    return plan


def expand_queries(query: str, *, max_variants: int = 3) -> list[str]:
    """兼容旧接口的轻量包装。"""

    out = _dedupe_keep_order([query])
    return out[: max(1, max_variants)]
