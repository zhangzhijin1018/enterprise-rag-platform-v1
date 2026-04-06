"""查询改写节点模块。

查询改写的作用是把用户自然语言问题转换成更适合检索的表达，
例如补充缺失上下文、压缩冗余口语、显式保留关键术语。
"""

from __future__ import annotations

from core.generation.prompts.templates import QUERY_REWRITE_SYSTEM
from core.orchestration.state import RAGState
from core.services.runtime import RAGRuntime


async def rewrite_query_node(state: RAGState, runtime: RAGRuntime) -> RAGState:
    """执行查询改写，并优先命中缓存。"""

    q = state.get("question") or ""
    cache_key = f"{state.get('query_type','')}:{q}"
    cached = runtime.cache.get_json("rewrite", cache_key)
    if cached and isinstance(cached.get("rewritten_query"), str):
        return {"rewritten_query": cached["rewritten_query"]}

    if not runtime.llm.enabled:
        # 离线模式下无法调用 LLM，就直接把原始问题作为改写结果。
        rewritten = q.strip()
        runtime.cache.set_json("rewrite", cache_key, {"rewritten_query": rewritten})
        return {"rewritten_query": rewritten}

    messages = [
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user", "content": f"Original question:\n{q}\n\nRewritten query:"},
    ]
    text, _ = await runtime.llm.complete(messages, temperature=0.0, max_tokens=128)
    # 这里只取首行，并限制最大长度，避免后续检索输入过长。
    rewritten = (text or q).strip().splitlines()[0][:512]
    runtime.cache.set_json("rewrite", cache_key, {"rewritten_query": rewritten})
    return {"rewritten_query": rewritten}
