"""查询规划与多路查询扩展模块。

本轮重构后，这里不再依赖固定业务分类，而是基于：

- `question`
- `resolved_query`
- `strategy_signals`

生成一份更适合企业问答场景的检索计划。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from core.generation.prompts.templates import QUERY_PLAN_SYSTEM
from core.services.runtime import RAGRuntime


@dataclass(slots=True)
class QueryPlan:
    """查询规划结果。

    这是 `build_query_plan()` 给 retrieval 准备的中间产物：
    - 一个主 query
    - 若干关键词 query
    - 若干子查询
    - 可选 HyDE query
    - 一组结构化过滤条件
    """

    resolved_query: str = ""
    rewritten_query: str = ""
    keyword_queries: list[str] = field(default_factory=list)
    multi_queries: list[str] = field(default_factory=list)
    hyde_query: str = ""
    structured_filters: dict[str, Any] = field(default_factory=dict)
    planning_summary: str = ""


_COMPARE_RE = re.compile(r"(.+?)(和|与|vs\.?|VS\.?|对比|比较)(.+)")
_ERROR_CODE_RE = re.compile(r"\b(ERR|ERROR|E)[-_]?\d+\b", re.IGNORECASE)
_QUOTED_TERM_RE = re.compile(r"[\"“”'`【】\[\]]([^\"“”'`【】\[\]]{2,40})[\"“”'`【】\[\]]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9._/-]{2,}|[\u4e00-\u9fff]{2,}")
_TIME_RE = re.compile(
    r"(今天|明天|后天|昨天|本周|下周|本月|上月|本季度|今年|去年|\d{1,2}月\d{1,2}[日号]?|\d{1,2}[日号])"
)
_SHIFT_RE = re.compile(r"(白班|夜班|早班|中班|晚班)")
_DEPARTMENT_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9_-]{2,30}(车间|部门|班组|小组|中心|事业部|工段|科室|仓库|产线|号线|线))"
)
_ENV_RE = re.compile(r"(生产|测试|预发|本地|线上|线下|docker|k8s|kubernetes|macos|linux|windows)", re.IGNORECASE)
_VERSION_RE = re.compile(r"(\b\d+\.\d+(\.\d+)?\b)")
_PERSON_RE = re.compile(r"(联系人|负责人|值班人|审批人|经理|主管|老师)\s*[:：]?\s*([\u4e00-\u9fff]{2,4})")
_FILLER_RE = re.compile(r"^(请问|想问一下|麻烦问下|帮我看下|帮我查下|咨询一下|我想知道)")
_LINE_RE = re.compile(r"(\d+号线|\d+线)")
_LEADING_TIME_PREFIX_RE = re.compile(r"^(今天|明天|后天|昨天|本周|下周|本月|上月|本季度|今年|去年)")
# 这一组 regex 主要服务于 query planning：
# - 从自然语言中抽 structured_filters
# - 识别 compare / error code / quoted term
# - 生成更适合 sparse 检索的 keyword query


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


def _extract_json_object(raw: str) -> dict[str, Any] | None:
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
    """把 LLM 输出规整成有限长度字符串列表。"""

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


def _coerce_filters(value: object) -> dict[str, Any]:
    """把 LLM 输出规整成结构化过滤条件。"""

    """把外部输入收敛成结构化过滤条件。"""

    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            continue
        k = key.strip()
        if not k:
            continue
        if isinstance(raw, (str, int, float, bool)):
            text = _normalize_query(str(raw))
            if text:
                out[k] = text
        elif isinstance(raw, list):
            vals = [str(item).strip() for item in raw if isinstance(item, (str, int, float))]
            vals = [item for item in vals if item]
            if vals:
                out[k] = vals
    return out


def _extract_keyword_candidates(question: str) -> list[str]:
    """提取适合稀疏检索的关键词候选。

    重点保留：
    - 错误码
    - 引号里的术语
    - 部门 / 班次 / 线别
    - 英文、数字、编号类 token
    """

    out = [m.group(0) for m in _ERROR_CODE_RE.finditer(question)]
    out.extend(m.group(1) for m in _QUOTED_TERM_RE.finditer(question))
    dept = _DEPARTMENT_RE.search(question)
    shift = _SHIFT_RE.search(question)
    line = _LINE_RE.search(question)
    if dept:
        out.append(dept.group(1))
    if shift:
        out.append(shift.group(1))
    if line:
        out.append(line.group(1))
    for token in _TOKEN_RE.findall(question):
        if re.search(r"[A-Za-z0-9._/-]", token):
            out.append(token)
    return _dedupe_keep_order(out)[:6]


def _extract_structured_filters(question: str) -> dict[str, Any]:
    """抽取可透传给检索层或后续结构化系统的过滤条件。

    这些 filters 既可能用于：
    - retrieval filter
    - metadata boost
    - 后续 explainability
    """

    filters: dict[str, Any] = {}
    time_match = _TIME_RE.search(question)
    dept_match = _DEPARTMENT_RE.search(question)
    shift_match = _SHIFT_RE.search(question)
    env_match = _ENV_RE.search(question)
    version_match = _VERSION_RE.search(question)
    line_match = _LINE_RE.search(question)
    person_match = _PERSON_RE.search(question)

    if time_match:
        filters["time"] = time_match.group(1)
    if dept_match:
        filters["department"] = _LEADING_TIME_PREFIX_RE.sub("", dept_match.group(1)).strip()
    if shift_match:
        filters["shift"] = shift_match.group(1)
    if env_match:
        filters["environment"] = env_match.group(1)
    if version_match:
        filters["version"] = version_match.group(1)
    if line_match:
        filters["line"] = line_match.group(1)
    if person_match:
        filters["person"] = person_match.group(2)
    return filters


def _build_rewritten_query(base_query: str, signals: dict[str, Any]) -> str:
    """生成主检索 query，默认偏保守，避免改写偏移。

    原则：
    - 精确实体和结构化事实查询尽量少改
    - 抽象问题才适度补“原因 / 排查 / 处理方法 / 步骤”等检索词
    """

    rewritten = _normalize_query(_FILLER_RE.sub("", base_query).strip())
    if not rewritten:
        return _normalize_query(base_query)

    if signals.get("likely_structured_lookup") or signals.get("has_precise_identifier"):
        return rewritten
    if re.search(r"(原因|排查|怎么处理|如何处理|解决|异常)", rewritten):
        return _normalize_query(f"{rewritten} 原因 排查 处理方法")
    if re.search(r"(制度|规范|要求|流程|步骤|SOP)", rewritten):
        return _normalize_query(f"{rewritten} 制度 规范 步骤")
    return rewritten


def _build_keyword_queries(base_query: str, signals: dict[str, Any], filters: dict[str, Any]) -> list[str]:
    """生成适合 BM25 / 倒排的关键词路线。

    这组 query 主要服务 sparse 检索，不追求语义完整，追求词面锚点更强。
    """

    candidates = _extract_keyword_candidates(base_query)
    if not signals.get("need_keyword_boost") and len(candidates) <= 1:
        return []

    out = list(candidates)
    department = filters.get("department")
    shift = filters.get("shift")
    time_value = filters.get("time")
    line = filters.get("line")
    if isinstance(department, str) and isinstance(shift, str):
        out.append(f"{department} {shift}")
    if isinstance(department, str) and isinstance(time_value, str):
        out.append(f"{department} {time_value}")
    if isinstance(line, str) and isinstance(time_value, str):
        out.append(f"{line} {time_value}")
    return _dedupe_keep_order(out)[:4]


def _build_multi_queries(base_query: str, signals: dict[str, Any]) -> list[str]:
    """根据问题结构生成多视角子查询。

    适合：
    - 对比问题
    - 原因 / 排查 / 处理问题
    - 方案 / 设计问题
    - 制度 / SOP / 流程问题
    """

    if not signals.get("need_sub_queries"):
        return []

    match = _COMPARE_RE.search(base_query)
    if match:
        left = _normalize_query(match.group(1))
        right = _normalize_query(match.group(3))
        return _dedupe_keep_order(
            [
                f"{left} 特点 适用场景",
                f"{right} 特点 适用场景",
                f"{left} {right} 区别",
            ]
        )[:3]

    if re.search(r"(原因|异常|报错|故障|失败)", base_query):
        return _dedupe_keep_order(
            [
                f"{base_query} 原因",
                f"{base_query} 排查步骤",
                f"{base_query} 处理方法",
            ]
        )[:3]

    if re.search(r"(方案|架构|设计|建设|选型|最佳实践)", base_query):
        return _dedupe_keep_order(
            [
                f"{base_query} 方案",
                f"{base_query} 适用场景",
                f"{base_query} 风险 注意事项",
            ]
        )[:3]

    if re.search(r"(制度|规范|流程|步骤|SOP)", base_query):
        return _dedupe_keep_order(
            [
                f"{base_query} 适用范围",
                f"{base_query} 执行步骤",
                f"{base_query} 注意事项",
            ]
        )[:3]

    return []


def _build_hyde_query(base_query: str, signals: dict[str, Any]) -> str:
    """仅在抽象解释类问题中启用 HyDE。

    当前 HyDE 的定位是：
    - 作为 dense 补充路线
    - 不默认总开
    - 避免在精确查询里把检索面弄得过宽
    """

    if not signals.get("need_hyde"):
        return ""
    return _normalize_query(
        f"这是一段企业知识说明，回答“{base_query}”，内容包含背景、原因、方案、步骤与注意事项。"
    )


def _heuristic_query_plan(
    question: str,
    *,
    resolved_query: str = "",
    strategy_signals: dict[str, Any] | None = None,
) -> QueryPlan:
    """规则版查询规划。

    当没有 LLM，或当前问题没必要用 LLM 规划时，就走这条规则路径。
    """

    signals = strategy_signals or {}
    question = _normalize_query(question)
    resolved = _normalize_query(resolved_query)
    base_query = resolved or question
    structured_filters = _extract_structured_filters(base_query)
    rewritten = _build_rewritten_query(base_query, signals)
    keyword_queries = _build_keyword_queries(base_query, signals, structured_filters)
    multi_queries = _build_multi_queries(base_query, signals)
    hyde_query = _build_hyde_query(base_query, signals)

    enabled_routes = ["direct_query"]
    if resolved:
        enabled_routes.append("resolved_query")
    if rewritten and rewritten != question and rewritten != resolved:
        enabled_routes.append("rewritten_query")
    if keyword_queries:
        enabled_routes.append("keyword_queries")
    if multi_queries:
        enabled_routes.append("multi_queries")
    if hyde_query:
        enabled_routes.append("hyde_query")

    return QueryPlan(
        resolved_query=resolved,
        rewritten_query=rewritten or question,
        keyword_queries=keyword_queries,
        multi_queries=multi_queries,
        hyde_query=hyde_query,
        structured_filters=structured_filters,
        planning_summary=f"启用路线：{', '.join(enabled_routes)}；过滤条件：{structured_filters or '无'}。",
    )


async def build_query_plan(
    runtime: RAGRuntime,
    *,
    question: str,
    resolved_query: str = "",
    strategy_signals: dict[str, Any] | None = None,
) -> QueryPlan:
    """生成一份查询计划。

    执行顺序：
    1. 命中缓存则直接返回；
    2. 无 LLM 时走规则版规划；
    3. 有 LLM 时尝试结构化生成，并用规则版做字段级兜底。
    """

    question = _normalize_query(question)
    resolved_query = _normalize_query(resolved_query)
    signals = dict(strategy_signals or {})
    cache_key = json.dumps(
        {
            "question": question,
            "resolved_query": resolved_query,
            "strategy_signals": signals,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    cached = runtime.cache.get_json("query_plan", cache_key)
    if isinstance(cached, dict):
        return QueryPlan(
            resolved_query=_normalize_query(str(cached.get("resolved_query") or resolved_query)),
            rewritten_query=str(cached.get("rewritten_query") or question),
            multi_queries=_coerce_str_list(cached.get("multi_queries"), limit=3),
            keyword_queries=_coerce_str_list(cached.get("keyword_queries"), limit=4),
            hyde_query=_normalize_query(str(cached.get("hyde_query") or "")),
            structured_filters=_coerce_filters(cached.get("structured_filters")),
            planning_summary=str(cached.get("planning_summary") or ""),
        )

    heuristic = _heuristic_query_plan(
        question,
        resolved_query=resolved_query,
        strategy_signals=signals,
    )
    if not runtime.llm.enabled:
        runtime.cache.set_json(
            "query_plan",
            cache_key,
            {
                "resolved_query": heuristic.resolved_query,
                "rewritten_query": heuristic.rewritten_query,
                "multi_queries": heuristic.multi_queries,
                "keyword_queries": heuristic.keyword_queries,
                "hyde_query": heuristic.hyde_query,
                "structured_filters": heuristic.structured_filters,
                "planning_summary": heuristic.planning_summary,
            },
        )
        return heuristic

    messages = [
        {"role": "system", "content": QUERY_PLAN_SYSTEM},
        {
            "role": "user",
            "content": (
                f"question: {question}\n"
                f"resolved_query: {resolved_query}\n"
                f"strategy_signals: {json.dumps(signals, ensure_ascii=False, sort_keys=True)}\n"
                "Return JSON only."
            ),
        },
    ]
    raw, _ = await runtime.llm.complete(
        messages,
        task="query_planning",
        temperature=0.0,
        max_tokens=640,
    )
    obj = _extract_json_object(raw) or {}

    plan = QueryPlan(
        resolved_query=_normalize_query(str(obj.get("resolved_query") or heuristic.resolved_query)),
        rewritten_query=_normalize_query(str(obj.get("rewritten_query") or heuristic.rewritten_query))
        or heuristic.rewritten_query
        or question,
        multi_queries=_coerce_str_list(obj.get("multi_queries"), limit=3) or heuristic.multi_queries,
        keyword_queries=(
            _coerce_str_list(obj.get("keyword_queries"), limit=4) or heuristic.keyword_queries
        ),
        hyde_query=_normalize_query(str(obj.get("hyde_query") or "")),
        structured_filters=_coerce_filters(obj.get("structured_filters")) or heuristic.structured_filters,
        planning_summary=_normalize_query(
            str(obj.get("planning_summary") or heuristic.planning_summary)
        ),
    )

    if not signals.get("need_hyde"):
        plan.hyde_query = ""
    if not signals.get("need_sub_queries"):
        plan.multi_queries = []
    if not signals.get("need_keyword_boost") and not heuristic.keyword_queries:
        plan.keyword_queries = []
    if not plan.resolved_query:
        plan.resolved_query = heuristic.resolved_query

    plan.multi_queries = _dedupe_keep_order(plan.multi_queries)[:3]
    plan.keyword_queries = _dedupe_keep_order(plan.keyword_queries)[:4]
    runtime.cache.set_json(
        "query_plan",
        cache_key,
        {
            "resolved_query": plan.resolved_query,
            "rewritten_query": plan.rewritten_query,
            "multi_queries": plan.multi_queries,
            "keyword_queries": plan.keyword_queries,
            "hyde_query": plan.hyde_query,
            "structured_filters": plan.structured_filters,
            "planning_summary": plan.planning_summary,
        },
    )
    return plan


def expand_queries(query: str, *, max_variants: int = 3) -> list[str]:
    """兼容旧接口的轻量包装。"""

    out = _dedupe_keep_order([query])
    return out[: max(1, max_variants)]
