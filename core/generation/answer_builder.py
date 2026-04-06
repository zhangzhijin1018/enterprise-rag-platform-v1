"""答案解析模块。

LLM 输出并不总是完全可靠，因此这里做两件事：
1. 从模型文本里解析 answer / confidence / reasoning / citations。
2. 对引用做白名单校验，只接受当前上下文里真实存在的 chunk_id。
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.generation.citation_formatter import Citation, chunk_to_citation
from core.retrieval.schemas import RetrievedChunk


_CIT_JSON = re.compile(r"CITATIONS_JSON:\s*(\[[\s\S]*?\])\s*$", re.MULTILINE)


def parse_llm_grounded_output(
    raw: str,
    contexts: list[RetrievedChunk],
) -> tuple[str, float, str, list[Citation]]:
    """解析 grounded prompt 的输出结果。"""

    # 如果模型格式不规范，至少先把原始文本作为 answer 保底返回。
    answer = raw.strip()
    confidence = 0.5
    reasoning = ""
    citations: list[Citation] = []

    # 按约定提取 `CONFIDENCE:` 段。
    m_conf = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", raw)
    if m_conf:
        confidence = float(m_conf.group(1))

    # 按约定提取 `REASONING_SUMMARY:` 段。
    m_rs = re.search(r"REASONING_SUMMARY:\s*(.+)", raw)
    if m_rs:
        reasoning = m_rs.group(1).strip()

    # 尽量把真正的答案正文从结构化标记中剥离出来。
    m_ans = re.search(r"ANSWER:\s*([\s\S]*?)(?=CONFIDENCE:|$)", raw)
    if m_ans:
        answer = m_ans.group(1).strip()

    # 优先解析结构化的 `CITATIONS_JSON`，因为它比正则抓文本引用更稳定。
    m_cj = _CIT_JSON.search(raw)
    if m_cj:
        try:
            arr = json.loads(m_cj.group(1))
            # 只允许引用当前上下文中的 chunk，避免模型“编造引用”。
            id_set = {c.chunk_id for c in contexts}
            for item in arr:
                cid = item.get("chunk_id") if isinstance(item, dict) else None
                if cid and cid in id_set:
                    ch = next(c for c in contexts if c.chunk_id == cid)
                    citations.append(chunk_to_citation(ch.metadata))
        except json.JSONDecodeError:
            # JSON 解析失败时静默降级到文本引用提取。
            pass

    if not citations:
        # 兼容另一种引用形式：`[CHUNK_ID:xxx]`。
        cited_ids = re.findall(r"\[CHUNK_ID:([^\]]+)\]", raw)
        ctx_map = {c.chunk_id: c for c in contexts}
        for cid in cited_ids:
            cid = cid.strip()
            if cid in ctx_map:
                citations.append(chunk_to_citation(ctx_map[cid].metadata))

    return answer, confidence, reasoning, citations
