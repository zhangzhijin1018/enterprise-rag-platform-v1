"""语义切块模块。

切块是 RAG 效果的关键环节之一：
- chunk 太大，召回容易混入无关信息；
- chunk 太小，上下文会碎片化，答案不完整；
- overlap 适度保留前后文，可降低切断句意带来的损失。
"""

from __future__ import annotations

import hashlib
import re
import uuid

from core.models.document import ChunkMetadata, Document, TextChunk

_PAGE_MARKER = re.compile(r"<!--\s*page:(\d+)\s*-->")


def _stable_chunk_id(doc_id: str, content: str, idx: int) -> str:
    """生成稳定 chunk_id。

    原理：
    - 用 `doc_id + chunk序号 + 内容前缀` 计算哈希。
    - 这样同一文档反复入库时，只要内容没明显变化，chunk_id 就尽量稳定。
    - 稳定 ID 对增量更新、引用追踪、测试断言都很重要。
    """

    h = hashlib.sha256(f"{doc_id}:{idx}:{content[:200]}".encode()).hexdigest()[:16]
    return f"{doc_id[:8]}-{idx}-{h}"


class SemanticChunker:
    """Split on markdown-style headings, page markers, and paragraph boundaries."""

    def __init__(
        self,
        max_chars: int = 1200,
        overlap: int = 150,
        min_chars: int = 80,
    ) -> None:
        self.max_chars = max_chars
        self.overlap = overlap
        self.min_chars = min_chars

    def chunk(self, doc: Document) -> list[TextChunk]:
        """把 Document 拆成多个 TextChunk。"""

        # 如果上游没给 doc_id，就现场生成一个，保证每个 chunk 都能追溯来源。
        doc_id = doc.doc_id or str(uuid.uuid4())
        # 先按标题分 section，尽量让同一主题内容留在同一个大段里。
        sections = self._split_sections(doc.content)
        chunks: list[TextChunk] = []
        idx = 0
        # `current_page` 用于继承最近一次出现的 PDF 页码标记。
        current_page: int | None = None
        for section_title, body in sections:
            # 每个 section 再按长度拆分，兼顾模型上下文限制和语义完整度。
            for piece in self._split_length(body):
                piece = piece.strip()
                # 太短的片段通常噪声大、信息密度低，这里直接跳过。
                if len(piece) < self.min_chars and section_title:
                    continue
                if not piece:
                    continue
                # PDF 解析器会把页码写成隐藏标记，切块时再把它抽出来放进 metadata。
                pm = _PAGE_MARKER.search(piece)
                page = int(pm.group(1)) if pm else current_page
                if pm:
                    piece = _PAGE_MARKER.sub("", piece).strip()
                    current_page = page
                # 每个 chunk 都要带完整 metadata，后续检索、重排、引用都依赖这些字段。
                meta = ChunkMetadata(
                    doc_id=doc_id,
                    chunk_id=_stable_chunk_id(doc_id, piece, idx),
                    source=doc.source,
                    title=doc.title,
                    page=page,
                    section=section_title or None,
                    extra={"mime_type": doc.mime_type or ""},
                )
                chunks.append(TextChunk(content=piece, metadata=meta))
                idx += 1
        return chunks

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

    def _split_length(self, text: str) -> list[str]:
        """按长度切分单个 section。"""

        text = text.strip()
        if not text:
            return []
        if len(text) <= self.max_chars:
            return [text]
        # 优先按段落切，而不是直接暴力截断，这样 chunk 的语义边界更自然。
        paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        out: list[str] = []
        cur = ""
        for p in paras:
            if len(cur) + len(p) + 2 <= self.max_chars:
                cur = f"{cur}\n\n{p}" if cur else p
            else:
                if cur:
                    # 当前缓存段超长时，再退化到滑窗切分。
                    out.extend(self._window_split(cur))
                if len(p) <= self.max_chars:
                    cur = p
                else:
                    out.extend(self._window_split(p))
                    cur = ""
        if cur:
            out.extend(self._window_split(cur))
        return out

    def _window_split(self, text: str) -> list[str]:
        """对超长文本做滑窗切分。

        示例：
        - `max_chars=1200`
        - `overlap=150`
        那么新 chunk 会保留上一个 chunk 的最后 150 个字符，
        用来减少“句子刚好在边界被切断”带来的信息损失。
        """

        if len(text) <= self.max_chars:
            return [text]
        chunks: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + self.max_chars)
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= n:
                break
            # 下一轮回退 overlap 个字符，构造窗口重叠区域。
            start = max(0, end - self.overlap)
        return chunks
