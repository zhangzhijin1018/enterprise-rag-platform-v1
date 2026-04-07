"""语义切块模块。

切块是 RAG 效果的关键环节之一：
- chunk 太大，召回容易混入无关信息；
- chunk 太小，上下文会碎片化，答案不完整；
- overlap 适度保留前后文，可降低切断句意带来的损失。

第三轮增强后，这个模块不再只生成“单层 chunk”，而是生成：

- `parent chunk`：更大、更完整，主要给 rerank / generation 使用
- `child chunk`：更小、更敏捷，主要给检索器使用

这样做的核心思路是：

1. 检索阶段更偏好较小的语义单元，方便精准命中；
2. 生成阶段更偏好较完整的上下文，避免答案被碎片化片段误导；
3. 所以让 child 负责召回，parent 负责回扩，是更稳的工程折中。
"""

from __future__ import annotations

import hashlib
import re
import uuid

from core.models.document import (
    CHILD_CHUNK_LEVEL,
    PARENT_CHUNK_LEVEL,
    ChunkMetadata,
    Document,
    TextChunk,
)

_PAGE_MARKER = re.compile(r"<!--\s*page:(\d+)\s*-->")


def _stable_chunk_id(doc_id: str, content: str, idx: int, *, prefix: str = "c") -> str:
    """生成稳定 chunk_id。

    原理：
    - 用 `doc_id + chunk序号 + 内容前缀` 计算哈希。
    - 这样同一文档反复入库时，只要内容没明显变化，chunk_id 就尽量稳定。
    - 稳定 ID 对增量更新、引用追踪、测试断言都很重要。
    """

    h = hashlib.sha256(f"{prefix}:{doc_id}:{idx}:{content[:200]}".encode()).hexdigest()[:16]
    return f"{prefix}-{doc_id[:8]}-{idx}-{h}"


class SemanticChunker:
    """按“section -> parent -> child”三层逻辑切块。

    参数设计说明：

    - `max_chars` / `overlap` 继续保留旧接口语义，对应 child chunk 的长度与重叠
    - `parent_max_chars` / `parent_overlap` 是分层增强后新增的 parent chunk 参数

    这样做可以尽量保证：

    - 旧调用方不需要改代码；
    - 新逻辑又能在内部生成两层 chunk 结构。
    """

    def __init__(
        self,
        max_chars: int = 1200,
        overlap: int = 150,
        min_chars: int = 80,
        parent_max_chars: int | None = None,
        parent_overlap: int | None = None,
    ) -> None:
        # `max_chars` / `overlap` 沿用旧语义，作为 child chunk 参数。
        self.max_chars = max_chars
        self.overlap = overlap
        self.min_chars = min_chars
        # parent chunk 默认比 child chunk 更大。
        # 这里不追求绝对最佳值，而是给一个稳妥、易理解的工程默认值：
        # - parent 至少不小于 child
        # - parent 默认大约是 child 的 2 倍左右
        self.parent_max_chars = max(parent_max_chars or max(max_chars * 2, max_chars + 400), max_chars)
        self.parent_overlap = parent_overlap or max(overlap, min(200, self.parent_max_chars // 5))

    def chunk(self, doc: Document) -> list[TextChunk]:
        """把 Document 拆成 parent + child 两层 TextChunk。

        返回顺序为：
        - 先 parent chunks
        - 后 child chunks

        这样调试时更直观，因为你打开 `chunks.jsonl` 时，
        更容易先看到“完整上下文块”，再看到对应的小片段。
        """

        # 如果上游没给 doc_id，就现场生成一个，保证每个 chunk 都能追溯来源。
        doc_id = doc.doc_id or str(uuid.uuid4())
        # 先按标题分 section，尽量让同一主题内容留在同一个大段里。
        sections = self._split_sections(doc.content)
        parent_chunks: list[TextChunk] = []
        child_chunks: list[TextChunk] = []
        parent_idx = 0
        child_idx = 0
        # `current_page` 用于继承最近一次出现的 PDF 页码标记。
        current_page: int | None = None
        for section_title, body in sections:
            # 先把 section 切成 parent chunks。
            for parent_piece in self._split_length(
                body,
                max_chars=self.parent_max_chars,
                overlap=self.parent_overlap,
            ):
                parent_piece = parent_piece.strip()
                if len(parent_piece) < self.min_chars and section_title:
                    continue
                if not parent_piece:
                    continue

                parent_piece, page, current_page = self._extract_page(parent_piece, current_page)
                parent_id = _stable_chunk_id(doc_id, parent_piece, parent_idx, prefix="p")
                parent_meta = ChunkMetadata(
                    doc_id=doc_id,
                    chunk_id=parent_id,
                    source=doc.source,
                    title=doc.title,
                    page=page,
                    section=section_title or None,
                    extra={
                        "mime_type": doc.mime_type or "",
                        "chunk_level": PARENT_CHUNK_LEVEL,
                        # parent 本身的“父 id”就定义为自己，
                        # 这样后续统一取 `parent_chunk_id` 时更简单。
                        "parent_chunk_id": parent_id,
                    },
                )
                parent_chunk = TextChunk(content=parent_piece, metadata=parent_meta)
                parent_chunks.append(parent_chunk)
                parent_idx += 1

                # 再把 parent chunk 继续切成 child chunks。
                # 这里 child 的职责是“更适合召回”，所以会更小、更细。
                raw_children = self._split_length(
                    parent_piece,
                    max_chars=self.max_chars,
                    overlap=self.overlap,
                )
                built_any_child = False
                for child_piece in raw_children:
                    child_piece = child_piece.strip()
                    if len(child_piece) < self.min_chars and len(parent_piece) > self.min_chars:
                        continue
                    if not child_piece:
                        continue
                    child_meta = ChunkMetadata(
                        doc_id=doc_id,
                        chunk_id=_stable_chunk_id(
                            doc_id,
                            f"{parent_id}:{child_piece}",
                            child_idx,
                            prefix="c",
                        ),
                        source=doc.source,
                        title=doc.title,
                        page=page,
                        section=section_title or None,
                        extra={
                            "mime_type": doc.mime_type or "",
                            "chunk_level": CHILD_CHUNK_LEVEL,
                            "parent_chunk_id": parent_id,
                        },
                    )
                    child_chunks.append(TextChunk(content=child_piece, metadata=child_meta))
                    child_idx += 1
                    built_any_child = True

                # 极端情况下，如果 parent 太短导致 child 全被过滤掉，
                # 仍然回退生成一个与 parent 等长的 child，避免“有 parent 没 child”。
                if not built_any_child:
                    child_meta = ChunkMetadata(
                        doc_id=doc_id,
                        chunk_id=_stable_chunk_id(
                            doc_id,
                            f"{parent_id}:{parent_piece}",
                            child_idx,
                            prefix="c",
                        ),
                        source=doc.source,
                        title=doc.title,
                        page=page,
                        section=section_title or None,
                        extra={
                            "mime_type": doc.mime_type or "",
                            "chunk_level": CHILD_CHUNK_LEVEL,
                            "parent_chunk_id": parent_id,
                        },
                    )
                    child_chunks.append(TextChunk(content=parent_piece, metadata=child_meta))
                    child_idx += 1

        return parent_chunks + child_chunks

    def _extract_page(
        self,
        piece: str,
        current_page: int | None,
    ) -> tuple[str, int | None, int | None]:
        """从片段里抽取页码标记，并返回清洗后的文本。"""

        pm = _PAGE_MARKER.search(piece)
        page = int(pm.group(1)) if pm else current_page
        if pm:
            piece = _PAGE_MARKER.sub("", piece).strip()
            current_page = page
        return piece, page, current_page

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """按 Markdown 标题切分 section。"""

        if not text.strip():
            return []
        parts: list[tuple[str, str]] = []
        lines = text.split("\n")
        current_title = ""
        buf: list[str] = []
        for line in lines:
            # 只识别 `#` 风格标题，这和 Markdown / HTML 转 Markdown 的输出形态相匹配。
            m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if m:
                if buf:
                    parts.append((current_title, "\n".join(buf).strip()))
                    buf = []
                current_title = m.group(2).strip()
            else:
                buf.append(line)
        if buf:
            parts.append((current_title, "\n".join(buf).strip()))
        if not parts:
            return [("", text)]
        return parts

    def _split_length(self, text: str, *, max_chars: int, overlap: int) -> list[str]:
        """按长度切分单个 section。

        这里抽成通用函数，是因为 parent 和 child 虽然职责不同，
        但底层的“段落优先 + 超长回退滑窗”算法是一致的。
        """

        text = text.strip()
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]
        # 优先按段落切，而不是直接暴力截断，这样 chunk 的语义边界更自然。
        paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        out: list[str] = []
        cur = ""
        for p in paras:
            if len(cur) + len(p) + 2 <= max_chars:
                cur = f"{cur}\n\n{p}" if cur else p
            else:
                if cur:
                    # 当前缓存段超长时，再退化到滑窗切分。
                    out.extend(self._window_split(cur, max_chars=max_chars, overlap=overlap))
                if len(p) <= max_chars:
                    cur = p
                else:
                    out.extend(self._window_split(p, max_chars=max_chars, overlap=overlap))
                    cur = ""
        if cur:
            out.extend(self._window_split(cur, max_chars=max_chars, overlap=overlap))
        return out

    def _window_split(self, text: str, *, max_chars: int, overlap: int) -> list[str]:
        """对超长文本做滑窗切分。

        示例：
        - `max_chars=1200`
        - `overlap=150`
        那么新 chunk 会保留上一个 chunk 的最后 150 个字符，
        用来减少“句子刚好在边界被切断”带来的信息损失。
        """

        if len(text) <= max_chars:
            return [text]
        chunks: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + max_chars)
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= n:
                break
            # 下一轮回退 overlap 个字符，构造窗口重叠区域。
            start = max(0, end - overlap)
        return chunks
