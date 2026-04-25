"""兜底策略模块。

这层的职责不是“想办法生成一个差不多的答案”，
而是在证据不足、召回为空、相关性过低时，给出一致、可解释的拒答结果。

对企业 RAG 来说，这比“硬答一个看起来像答案的东西”更重要。
"""

from __future__ import annotations

from core.orchestration.state import RAGState


def empty_retrieval_refusal(state: RAGState) -> RAGState:
    """构造统一的空召回拒答结果。

    统一封装成函数，而不是散落在各节点里手写字典，有两个好处：
    - API 层看到的 refusal 语义更稳定
    - 后续如果要统一调整拒答文案、字段或 reason code，改动面更小
    """

    return {
        "refusal": True,
        "refusal_reason": "no_relevant_chunks",
        "answer": "根据当前知识库检索结果，未找到足够相关的文档片段，无法安全作答。",
        "confidence": 0.0,
        "reasoning_summary": "检索为空或分数过低，触发拒答策略。",
        "citations": [],
        "grounding_ok": False,
    }
