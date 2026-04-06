"""查询分析节点模块。负责对问题做轻量分类，为后续改写和策略分支提供上下文。"""

from __future__ import annotations

import re

from core.orchestration.state import RAGState


_CODE_RE = re.compile(r"\b(ERR|ERROR|E)[-_]?\d+\b", re.IGNORECASE)


async def analyze_query_node(state: RAGState) -> RAGState:
    q = state.get("question") or ""
    qtype = "general"
    if _CODE_RE.search(q):
        qtype = "error_code"
    elif re.search(r"流程|步骤|SOP|怎么|如何", q):
        qtype = "procedure"
    return {"query_type": qtype, "multi_queries": []}
