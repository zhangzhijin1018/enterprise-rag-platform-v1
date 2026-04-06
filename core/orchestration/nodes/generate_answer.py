"""答案生成节点模块。

它负责拿到最终重排结果后：
1. 做最后一次“是否足够可靠”的门控；
2. 组织 grounded prompt；
3. 调用 LLM；
4. 解析结构化答案与引用。
"""

from __future__ import annotations

from core.generation.answer_builder import parse_llm_grounded_output
from core.generation.context_format import format_context_blocks
from core.generation.prompts.templates import GROUNDED_ANSWER_SYSTEM
from core.orchestration.state import RAGState
from core.retrieval.schemas import RetrievedChunk
from core.services.runtime import RAGRuntime


async def generate_answer_node(state: RAGState, runtime: RAGRuntime) -> RAGState:
    """执行回答生成节点。"""

    rq = state.get("rewritten_query") or state.get("question") or ""
    ctx_raw = state.get("reranked_hits") or []
    contexts = [RetrievedChunk.model_validate(x) for x in ctx_raw]
    if not contexts:
        # 没有上下文时直接拒答，避免模型凭空编造答案。
        return {
            "answer": "上下文不足，已拒答。",
            "confidence": 0.0,
            "reasoning_summary": "无可用上下文。",
            "citations": [],
            "refusal": True,
            "refusal_reason": "empty_context",
            "grounding_ok": False,
        }

    max_score = max(c.score for c in contexts)
    if max_score < runtime.settings.min_rerank_score:
        # 即使有上下文，也要保证它和问题足够相关。
        return {
            "answer": "检索到的内容与问题相关性不足，无法可靠作答。",
            "confidence": 0.15,
            "reasoning_summary": "重排分数低于阈值。",
            "citations": [],
            "refusal": True,
            "refusal_reason": "low_relevance",
            "grounding_ok": False,
        }

    # 把召回上下文拼成带 chunk_id 的块，便于模型进行 grounded generation。
    ctx_text = format_context_blocks(contexts)
    user = f"QUESTION:\n{rq}\n\nCONTEXT:\n{ctx_text}"
    messages = [
        {"role": "system", "content": GROUNDED_ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]
    raw, _ = await runtime.llm.complete(messages, temperature=0.1, max_tokens=1024)
    # 统一解析 LLM 输出，抽取 answer / confidence / citations。
    answer, conf, reasoning, citations = parse_llm_grounded_output(raw, contexts)

    return {
        "answer": answer,
        "confidence": conf,
        "reasoning_summary": reasoning,
        "citations": [c.model_dump(mode="json") for c in citations],
        "refusal": False,
        "refusal_reason": "",
    }
