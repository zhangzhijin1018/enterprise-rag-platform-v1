"""澄清判定节点模块。

这个模块解决的是一个经常被忽略、但对 RAG 稳定性影响非常大的问题：

    并不是所有用户问题都适合直接进入检索。

如果用户问题缺少关键实体、报错信息、版本、环境或比较对象，
那么后面的 query rewrite、multi-query、dense retrieval 只会把歧义进一步放大。
因此这里把“是否需要先向用户追问”单独抽成一个显式节点。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

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


_ERROR_WORD_RE = re.compile(r"(报错|错误|异常|失败|问题|不生效|无法|不工作)")
_ERROR_DETAIL_RE = re.compile(
    r"(\b(ERR|ERROR|E)[-_]?\d+\b|traceback|exception|stack trace|[A-Za-z]+Error|[A-Za-z]+Exception)",
    re.IGNORECASE,
)
_DEICTIC_COMPARE_RE = re.compile(r"(它|这个|那个|前者|后者).*(区别|差异|哪个好|谁更好|怎么选)")
_DEICTIC_RE = re.compile(r"^(这个|那个|它|这个问题|这个报错).*(怎么|如何|怎么办|咋办)")
_VERSION_RE = re.compile(r"(\b\d+\.\d+(\.\d+)?\b|版本|version)", re.IGNORECASE)
_ENV_RE = re.compile(r"(生产|测试|本地|线上|线下|docker|k8s|kubernetes|macos|linux|windows)", re.IGNORECASE)
_PRODUCT_RE = re.compile(
    r"\b(Milvus|Zilliz|Redis|Docker|Kubernetes|K8s|MySQL|PostgreSQL|OpenAI|LangGraph|RAG)\b",
    re.IGNORECASE,
)


def _normalize_question(question: str) -> str:
    """统一收敛用户问题文本。"""

    return re.sub(r"\s+", " ", question).strip()


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


def _coerce_list(value: object) -> list[str]:
    """把外部输入尽量转成字符串列表。"""

    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
    return out


def _heuristic_clarify(question: str) -> ClarifyDecision:
    """规则版澄清判定。

    设计目标不是覆盖所有语言现象，而是优先拦住最容易误召回的高风险问题。
    """

    q = _normalize_question(question)
    if len(q) <= 4:
        return ClarifyDecision(
            need_clarify=True,
            missing_slots=["target_object"],
            clarify_question="请补充你具体想问的对象、模块或问题背景，我再继续帮你检索和分析。",
            clarify_reason="问题过短，缺少足够的检索锚点。",
        )

    if _ERROR_WORD_RE.search(q) and not _ERROR_DETAIL_RE.search(q):
        if _DEICTIC_RE.search(q) or len(q) <= 20:
            return ClarifyDecision(
                need_clarify=True,
                missing_slots=["error_code", "log_snippet", "runtime_context"],
                clarify_question=(
                    "请补充具体错误码、报错日志关键片段，以及问题发生的环境"
                    "（例如本地、测试、生产，或 Docker / K8s）。"
                ),
                clarify_reason="当前问题像排障请求，但缺少错误码、日志或运行环境，直接检索容易误召回。",
            )

    if _DEICTIC_COMPARE_RE.search(q):
        return ClarifyDecision(
            need_clarify=True,
            missing_slots=["comparison_targets"],
            clarify_question="请明确你要比较的两个对象分别是什么，我再帮你做差异分析。",
            clarify_reason="比较关系存在，但比较对象使用了模糊指代，系统无法可靠检索。",
        )

    if re.search(r"(怎么部署|如何部署|部署方案|架构方案|怎么搭|如何搭)", q):
        has_product = bool(_PRODUCT_RE.search(q))
        has_version = bool(_VERSION_RE.search(q))
        has_env = bool(_ENV_RE.search(q))
        if not has_product or (not has_version and not has_env and len(q) <= 18):
            missing: list[str] = []
            if not has_product:
                missing.append("product_name")
            if not has_version:
                missing.append("version")
            if not has_env:
                missing.append("deployment_environment")
            return ClarifyDecision(
                need_clarify=True,
                missing_slots=missing,
                clarify_question=(
                    "请补充具体产品、版本和部署环境"
                    "（例如单机 / 分布式、本地 / 云上、Docker / K8s），"
                    "这样我才能给出更准确的部署建议。"
                ),
                clarify_reason="部署类问题缺少产品、版本或环境约束，直接进入 RAG 会得到过于泛化的答案。",
            )

    return ClarifyDecision()


async def clarify_query_node(state: RAGState, runtime: RAGRuntime) -> RAGState:
    """判断当前问题是否应该先澄清再检索。"""

    question = _normalize_question(state.get("question") or "")
    cache_key = f"{state.get('query_type','')}:{question}"
    cached = runtime.cache.get_json("clarify", cache_key)
    if isinstance(cached, dict):
        return {
            "need_clarify": bool(cached.get("need_clarify")),
            "missing_slots": _coerce_list(cached.get("missing_slots")),
            "clarify_question": str(cached.get("clarify_question") or ""),
            "clarify_reason": str(cached.get("clarify_reason") or ""),
        }

    heuristic = _heuristic_clarify(question)
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
                f"query_type: {state.get('query_type', 'general')}\n"
                "Return JSON only."
            ),
        },
    ]
    raw, _ = await runtime.llm.complete(messages, temperature=0.0, max_tokens=256)
    obj = _extract_json_object(raw) or {}

    need_clarify = bool(obj.get("need_clarify"))
    clarify_question = str(obj.get("clarify_question") or "").strip()
    clarify_reason = str(obj.get("clarify_reason") or "").strip()
    missing_slots = _coerce_list(obj.get("missing_slots"))

    # 模型如果说需要澄清，但没有给出可用追问句，则回退到规则版结果。
    if need_clarify and not clarify_question:
        need_clarify = heuristic.need_clarify
        clarify_question = heuristic.clarify_question
        clarify_reason = heuristic.clarify_reason
        missing_slots = heuristic.missing_slots

    # 如果规则已经命中明显高风险模糊问题，就优先相信规则版，不轻易放过去。
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
        or "当前问题信息不足，请补充更具体的对象、上下文或错误信息后再继续。"
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
