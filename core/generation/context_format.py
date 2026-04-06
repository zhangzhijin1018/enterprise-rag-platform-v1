"""上下文格式化模块。

生成前会把检索结果拼成带 metadata 的上下文块，
让 LLM 能看到来源、页码、section，并用 chunk_id 显式引用。
"""

from __future__ import annotations

from core.retrieval.schemas import RetrievedChunk


def format_context_blocks(contexts: list[RetrievedChunk]) -> str:
    """把多个检索片段拼成 prompt 中的上下文文本。"""

    parts: list[str] = []
    for c in contexts:
        meta = c.metadata
        sec = meta.section or ""
        page = meta.page or ""
        # 示例：
        # [CHUNK_ID:abc] title=密码重置 SOP source=sop.md page=2 section=常见错误
        # 这样模型既能看内容，也能产出可追溯引用。
        parts.append(
            f"[CHUNK_ID:{c.chunk_id}] title={meta.title} source={meta.source} "
            f"page={page} section={sec}\n{c.content}\n"
        )
    return "\n\n".join(parts)
