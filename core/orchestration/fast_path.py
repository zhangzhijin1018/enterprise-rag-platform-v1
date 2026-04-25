"""快速问答前置链路模块。

这层专门对应图里的“Redis 检索系统 + MySQL 检索系统”：

1. 先查 Redis 热点答案缓存
2. 再查 MySQL FAQ + BM25
3. 还命不中，再进入完整 RAG

它的价值不是替代 RAG，而是把“高频、稳定、结构化”的问题优先消化掉，
减少向量检索、重排和 LLM 生成的成本。
"""

from __future__ import annotations

from core.orchestration.state import RAGState
from core.services.runtime import RAGRuntime


def _normalize_question(question: str) -> str:
    """归一化问题文本，作为 Redis 命中的稳定键。"""

    return " ".join(question.strip().lower().split())


def _faq_entry_id_from_source_id(source_id: str) -> int | None:
    """从 `faq:123` 这类 source_id 中提取 FAQ 主键。"""

    raw = source_id.strip()
    if not raw.startswith("faq:"):
        return None
    tail = raw.split(":", 1)[1].strip()
    return int(tail) if tail.isdigit() else None


def _record_faq_hit(runtime: RAGRuntime, source_id: str) -> None:
    """在快速通道命中 FAQ 时累加命中次数。

    这里做成一个尽量弱依赖的辅助函数：

    - 运行时没有 `faq_store` 时也不报错
    - 命中来源不是 FAQ 时直接跳过
    """

    entry_id = _faq_entry_id_from_source_id(source_id)
    faq_store = getattr(runtime, "faq_store", None)
    if entry_id is None or faq_store is None:
        return
    record_hit = getattr(faq_store, "record_hit", None)
    if callable(record_hit):
        record_hit(entry_id)


def _build_fast_path_state(
    *,
    question: str,
    answer: str,
    confidence: float,
    source: str,
    source_id: str,
    title: str,
    section: str | None = None,
) -> RAGState:
    """把快速通道命中结果包装成统一 `RAGState`。

    这里虽然不是 RAG 召回出来的 chunk，但仍然构造了一个伪 chunk：

    - 前端仍然能统一展示来源
    - `/chat` 的返回结构完全不用改
    - 后续日志和评测也能知道答案来自哪条快速通道
    """

    retrieved = {
        "chunk_id": source_id,
        "score": float(confidence),
        "content": answer,
        "metadata": {
            "doc_id": source_id,
            "chunk_id": source_id,
            "source": source,
            "title": title,
            "section": section,
        },
    }
    citation = {
        "doc_id": source_id,
        "chunk_id": source_id,
        "title": title,
        "source": source,
        "section": section,
    }
    return {
        "question": question,
        "answer": answer,
        "confidence": float(confidence),
        "reasoning_summary": f"fast_path:{source}",
        "citations": [citation],
        "reranked_hits": [retrieved],
        "refusal": False,
        "refusal_reason": "",
        "grounding_ok": True,
        "fast_path_source": source,
    }


async def try_fast_path_answer(runtime: RAGRuntime, question: str) -> RAGState | None:
    """尝试通过 Redis / MySQL FAQ 快速返回答案。

    这层的定位是“先消化高频、稳定、标准化的问题”，
    而不是替代后面的完整 RAG。
    """

    normalized = _normalize_question(question)
    cached = runtime.cache.get_json("answer", normalized)
    if isinstance(cached, dict) and cached.get("answer"):
        # Redis 层缓存的是“最终答案”，命中后直接返回。
        source_id = str(cached.get("source_id") or "cache:answer")
        _record_faq_hit(runtime, source_id)
        return _build_fast_path_state(
            question=question,
            answer=str(cached["answer"]),
            confidence=float(cached.get("confidence") or 0.95),
            source="redis_answer_cache",
            source_id=source_id,
            title=str(cached.get("title") or "Redis 热点答案缓存"),
            section=str(cached.get("section") or "cache"),
        )

    matches = runtime.faq_retriever.search(question, top_k=1)
    if not matches:
        return None
    best = matches[0]
    # FAQ 虽然命中了，但仍要过一个阈值，避免把边缘相似问题误当成标准问答。
    if best.confidence < runtime.settings.faq_bm25_threshold:
        return None

    source_id = f"faq:{best.entry.entry_id}"
    _record_faq_hit(runtime, source_id)
    payload = {
        "answer": best.entry.answer,
        "confidence": best.confidence,
        "source_id": source_id,
        "title": best.entry.question,
        "section": best.entry.category or "mysql_faq",
    }
    runtime.cache.set_json(
        "answer",
        normalized,
        payload,
        ttl_sec=runtime.settings.answer_cache_ttl_sec,
    )
    return _build_fast_path_state(
        question=question,
        answer=best.entry.answer,
        confidence=best.confidence,
        source="mysql_faq",
        source_id=source_id,
        title=best.entry.question,
        section=best.entry.category or "mysql_faq",
    )
