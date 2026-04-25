"""本地受限模式回答执行模块。

当上下文不允许出域，但系统又希望基于“已授权证据”给出一个可解释回答时，
这里负责构造一个本地可解析的 grounded 输出。
"""

from __future__ import annotations

import json
import re

from core.retrieval.schemas import RetrievedChunk


def _compact_text(text: str, limit: int = 80) -> str:
    """把文本压缩成更短的可展示片段。"""

    value = re.sub(r"\s+", " ", text).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _first_sentence(text: str) -> str:
    """尽量抽取首句，作为本地受限回答的证据摘要。"""

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""
    parts = re.split(r"(?<=[。！？.!?])\s+", normalized, maxsplit=1)
    return parts[0].strip() or _compact_text(normalized)


def build_local_grounded_output(
    *,
    question: str,
    contexts: list[RetrievedChunk],
    conflict_summary: str | None = None,
) -> str:
    """基于受限上下文构造可解析的本地占位答案。

    这里返回的不是自由文本，而是和外部 LLM 输出同样可解析的结构：
    - `ANSWER`
    - `CONFIDENCE`
    - `REASONING_SUMMARY`
    - `CITATIONS_JSON`

    这样上层可以继续复用同一套 `parse_llm_grounded_output()`，
    不需要为本地降级再写另一套解析分支。
    """

    cited = contexts[: min(2, len(contexts))]
    if not cited:
        # 没有任何可用证据时，也返回可解析结构，保持输出协议稳定。
        return (
            "ANSWER: 当前没有可用于本地受限模式的上下文。\n"
            "CONFIDENCE: 0.0\n"
            "REASONING_SUMMARY: 本地受限模式没有可用证据。\n"
            "CITATIONS_JSON: []"
        )

    answer_parts: list[str] = []
    for ctx in cited:
        sentence = _first_sentence(ctx.content) or _compact_text(ctx.content)
        answer_parts.append(f"{sentence} [CHUNK_ID:{ctx.chunk_id}]")

    preface = "根据当前已授权的本地受限上下文"
    if conflict_summary:
        # 冲突摘要不是直接替代答案，而是作为受限模式下的治理提示拼进前缀里。
        preface += f"，并结合治理提示“{_compact_text(conflict_summary, 60)}”"
    answer = preface + "，" + "；".join(answer_parts)
    citations = [{"chunk_id": ctx.chunk_id} for ctx in cited]
    return (
        f"ANSWER: {answer}\n"
        "CONFIDENCE: 0.62\n"
        "REASONING_SUMMARY: 使用本地受限模式，根据已授权上下文直接生成答案。\n"
        f"CITATIONS_JSON: {json.dumps(citations, ensure_ascii=False)}"
    )
