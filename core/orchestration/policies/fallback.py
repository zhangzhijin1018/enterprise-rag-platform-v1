"""兜底策略模块。定义空召回、低相关等场景下的统一拒答策略。"""

from __future__ import annotations

from core.orchestration.state import RAGState


def empty_retrieval_refusal(state: RAGState) -> RAGState:
    return {
        "refusal": True,
        "refusal_reason": "no_relevant_chunks",
        "answer": "根据当前知识库检索结果，未找到足够相关的文档片段，无法安全作答。",
        "confidence": 0.0,
        "reasoning_summary": "检索为空或分数过低，触发拒答策略。",
        "citations": [],
        "grounding_ok": False,
    }
