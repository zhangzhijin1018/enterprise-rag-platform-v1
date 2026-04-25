"""上下文格式化模块。

这一层解决的是一个很实际的问题：

- 检索器返回的是一组 `RetrievedChunk`
- 模型真正接收的是一段 prompt 文本

所以这里负责把“结构化检索结果”压缩并翻译成“适合给模型看的证据上下文”。
"""

from __future__ import annotations

from collections import defaultdict

from core.retrieval.schemas import RetrievedChunk


def select_contexts_for_prompt(
    contexts: list[RetrievedChunk],
    *,
    max_docs: int,
    max_chunks_per_doc: int,
    max_chars: int,
) -> list[RetrievedChunk]:
    """为 prompt 选择更紧凑的上下文。

    策略：
    - 先按当前排序顺序遍历，保留高分上下文优先级
    - 每个文档限制 chunk 数，避免同文档重复占满 token
    - 全局控制文档数和累计字符数
    - 若 section_path 重复，则优先保留首个 chunk

    这一步本质上是在做“生成前压缩”，
    目标不是找出所有相关 chunk，而是找出“最值得送进 prompt 的那一小组证据”。
    """

    selected: list[RetrievedChunk] = []
    doc_counts: dict[str, int] = defaultdict(int)
    section_seen: dict[str, set[str]] = defaultdict(set)
    seen_docs: set[str] = set()
    used_chars = 0

    for item in contexts:
        doc_id = item.metadata.doc_id
        section_path = (item.metadata.extra_text("section_path") or item.metadata.section or "").strip()
        # 先限制文档总数，避免 prompt 被少数低价值长文档拖得过长。
        if doc_id not in seen_docs and len(seen_docs) >= max(1, max_docs):
            continue
        # 再限制单文档 chunk 数，避免同一文档把上下文窗口挤满。
        if doc_counts[doc_id] >= max(1, max_chunks_per_doc):
            continue
        # 相同 section_path 只保留第一个，降低重复段落对生成的干扰。
        if section_path and section_path in section_seen[doc_id]:
            continue
        estimated_chars = len(item.content) + 160
        # 用一个保守估算控制总字符数，目标是把 token 成本稳定压在可控范围。
        if selected and used_chars + estimated_chars > max(1, max_chars):
            continue
        selected.append(item)
        seen_docs.add(doc_id)
        doc_counts[doc_id] += 1
        if section_path:
            section_seen[doc_id].add(section_path)
        used_chars += estimated_chars
    return selected or contexts[:1]


def format_context_blocks(contexts: list[RetrievedChunk]) -> str:
    """把多个检索片段拼成 prompt 中的上下文文本。

    输出格式会显式带上：
    - `CHUNK_ID`
    - 标题 / 来源
    - 页码 / section

    这样模型在回答时既能利用内容，也能产出可追溯引用。

    这里之所以显式写出 `CHUNK_ID`，是因为后续答案解析会按这个 id 回收引用，
    从而避免模型只给模糊的“根据某文档”式引用。
    """

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
