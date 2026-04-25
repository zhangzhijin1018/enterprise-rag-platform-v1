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
from dataclasses import dataclass

from core.models.document import (
    CHUNK_SEMANTIC_LIST_KEYS,
    CHUNK_SEMANTIC_SCALAR_KEYS,
    CHILD_CHUNK_LEVEL,
    ENTERPRISE_METADATA_KEYS,
    PARENT_CHUNK_LEVEL,
    ChunkMetadata,
    Document,
    TextChunk,
    normalize_enterprise_metadata_value,
)

_PAGE_MARKER = re.compile(r"<!--\s*page:(\d+)\s*-->")
_KEYWORD_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_/-]{2,30}")


def _stable_chunk_id(doc_id: str, content: str, idx: int, *, prefix: str = "c") -> str:
    """生成稳定 chunk_id。

    原理：
    - 用 `doc_id + chunk序号 + 内容前缀` 计算哈希。
    - 这样同一文档反复入库时，只要内容没明显变化，chunk_id 就尽量稳定。
    - 稳定 ID 对增量更新、引用追踪、测试断言都很重要。
    """

    h = hashlib.sha256(f"{prefix}:{doc_id}:{idx}:{content[:200]}".encode()).hexdigest()[:16]
    return f"{prefix}-{doc_id[:8]}-{idx}-{h}"


@dataclass(frozen=True)
class ChunkProfile:
    """按文档类型裁剪出来的切块参数。

    这里不引入多套 chunker，而是保留统一切块器，
    再按文档类型做轻量 profile 调整，平衡：
    - 行为一致性
    - 维护成本
    - 不同文件类型的结构差异

    也就是说：
    - 核心切块算法只有一套
    - 不同文件类型只是在长度、重叠、页级切分策略上做轻量调参
    """

    child_max_chars: int
    child_overlap: int
    parent_max_chars: int
    parent_overlap: int
    min_chars: int
    split_pdf_by_page: bool = False


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

        这样调试和排障时更直观：
        - 先看到“完整上下文块”
        - 再看到对应的小片段
        不管底层最终存到 Milvus 还是其他后端，这个顺序都更利于理解 parent / child 关系。
        """

        profile = self._resolve_profile(doc)
        # 如果上游没给 doc_id，就现场生成一个，保证每个 chunk 都能追溯来源。
        doc_id = doc.doc_id or str(uuid.uuid4())
        # 先按标题分 section，尽量让同一主题内容留在同一个大段里。
        sections = self._split_sections(doc.content, profile=profile)
        parent_chunks: list[TextChunk] = []
        child_chunks: list[TextChunk] = []
        parent_idx = 0
        child_idx = 0
        # `current_page` 用于继承最近一次出现的 PDF 页码标记。
        current_page: int | None = None
        for section_title, section_path, section_level, body in sections:
            # 先把 section 切成 parent chunks。
            for parent_piece in self._split_length(
                body,
                max_chars=profile.parent_max_chars,
                overlap=profile.parent_overlap,
            ):
                parent_piece = parent_piece.strip()
                if len(parent_piece) < profile.min_chars and section_title:
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
                    extra=self._build_chunk_extra(
                        doc,
                        content=parent_piece,
                        section_title=section_title,
                        section_path=section_path,
                        section_level=section_level,
                        chunk_level=PARENT_CHUNK_LEVEL,
                        parent_chunk_id=parent_id,
                    ),
                )
                parent_chunk = TextChunk(content=parent_piece, metadata=parent_meta)
                parent_chunks.append(parent_chunk)
                parent_idx += 1

                # 再把 parent chunk 继续切成 child chunks。
                # 这里 child 的职责是“更适合召回”，所以会更小、更细。
                raw_children = self._split_length(
                    parent_piece,
                    max_chars=profile.child_max_chars,
                    overlap=profile.child_overlap,
                )
                built_any_child = False
                for child_piece in raw_children:
                    child_piece = child_piece.strip()
                    if len(child_piece) < profile.min_chars and len(parent_piece) > profile.min_chars:
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
                        extra=self._build_chunk_extra(
                            doc,
                            content=child_piece,
                            section_title=section_title,
                            section_path=section_path,
                            section_level=section_level,
                            chunk_level=CHILD_CHUNK_LEVEL,
                            parent_chunk_id=parent_id,
                        ),
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
                        extra=self._build_chunk_extra(
                            doc,
                            content=parent_piece,
                            section_title=section_title,
                            section_path=section_path,
                            section_level=section_level,
                            chunk_level=CHILD_CHUNK_LEVEL,
                            parent_chunk_id=parent_id,
                        ),
                    )
                    child_chunks.append(TextChunk(content=parent_piece, metadata=child_meta))
                    child_idx += 1

        return parent_chunks + child_chunks

    def _resolve_profile(self, doc: Document) -> ChunkProfile:
        """根据文档类型返回切块 profile。

        这里的核心思想不是“每种文件都造一个专用 chunker”，而是：

        - `CSV` 更结构化，chunk 要更小、更紧
        - `PPTX` 一页内容通常较短，适合更紧凑的 slide 级切块
        - `PDF` 更依赖页码语义，所以额外开启按页拆 section
        - `TXT` 缺少显式结构，参数上更保守，避免切得太碎
        """

        doc_type = str(doc.metadata.get("doc_type") or "").strip().lower()
        mime_type = str(doc.mime_type or "").strip().lower()

        profile = ChunkProfile(
            child_max_chars=self.max_chars,
            child_overlap=self.overlap,
            parent_max_chars=self.parent_max_chars,
            parent_overlap=self.parent_overlap,
            min_chars=self.min_chars,
        )

        if doc_type == "csv" or mime_type == "text/csv":
            return ChunkProfile(
                child_max_chars=min(profile.child_max_chars, 220),
                child_overlap=0,
                parent_max_chars=min(profile.parent_max_chars, 420),
                parent_overlap=0,
                min_chars=min(profile.min_chars, 20),
            )
        if doc_type == "pptx" or "presentationml.presentation" in mime_type:
            return ChunkProfile(
                child_max_chars=min(profile.child_max_chars, 420),
                child_overlap=min(profile.child_overlap, 40),
                parent_max_chars=min(profile.parent_max_chars, 960),
                parent_overlap=min(profile.parent_overlap, 80),
                min_chars=min(profile.min_chars, 40),
            )
        if doc_type == "pdf" or mime_type == "application/pdf":
            return ChunkProfile(
                child_max_chars=min(profile.child_max_chars, 900),
                child_overlap=min(profile.child_overlap, 120),
                parent_max_chars=min(profile.parent_max_chars, 1800),
                parent_overlap=min(profile.parent_overlap, 180),
                min_chars=profile.min_chars,
                split_pdf_by_page=True,
            )
        if doc_type in {"txt", "text"} or mime_type == "text/plain":
            return ChunkProfile(
                child_max_chars=min(profile.child_max_chars, 1000),
                child_overlap=min(profile.child_overlap, 100),
                parent_max_chars=min(profile.parent_max_chars, 1800),
                parent_overlap=min(profile.parent_overlap, 160),
                min_chars=profile.min_chars,
            )
        return profile

    def _build_chunk_extra(
        self,
        doc: Document,
        *,
        content: str,
        section_title: str,
        section_path: str,
        section_level: int,
        chunk_level: str,
        parent_chunk_id: str,
    ) -> dict[str, str | list[str]]:
        """把文档级 retrieval metadata 复制到 chunk extra，便于后续过滤。

        这里是文档级治理语义真正下沉到 chunk 的关键点：
        - 文档级 metadata 决定“属于什么知识对象”
        - chunk 级 semantic metadata 决定“这一小段在文档里扮演什么角色”

        如果这一步不做，下游会出现两个典型问题：
        - 检索命中 chunk 时，无法直接按 ACL / 分类 / 业务域做过滤和 boost
        - 引用展示时，只能看到片段内容，看不到足够的来源和治理上下文
        """

        extra: dict[str, str | list[str]] = {
            "mime_type": doc.mime_type or "",
            "chunk_level": chunk_level,
            "parent_chunk_id": parent_chunk_id,
        }
        for key in ENTERPRISE_METADATA_KEYS:
            normalized = normalize_enterprise_metadata_value(key, doc.metadata.get(key))
            if normalized is None:
                continue
            extra[key] = normalized
        # 再补一层 chunk 局部语义，供 retrieval boost、citation 和 explainability 使用。
        semantic_fields = self._build_chunk_semantic_metadata(
            doc=doc,
            content=content,
            section_title=section_title,
            section_path=section_path,
            section_level=section_level,
        )
        for key in CHUNK_SEMANTIC_SCALAR_KEYS + CHUNK_SEMANTIC_LIST_KEYS:
            value = semantic_fields.get(key)
            if value in (None, "", []):
                continue
            extra[key] = value
        return extra

    def _build_chunk_semantic_metadata(
        self,
        *,
        doc: Document,
        content: str,
        section_title: str,
        section_path: str,
        section_level: int,
    ) -> dict[str, str | list[str]]:
        """补齐 chunk 局部语义，供后续 metadata boost 与引用使用。

        这一步的目的不是做重语义理解，而是尽量把“局部结构信号”显式化：
        - section_path / section_type
        - 是否是步骤、表格、联系人、版本信号
        - 一组轻量 topic keywords

        换句话说，这里做的是“便宜但够用的 chunk 语义标注”，
        不是调用大模型去做昂贵的深层理解。
        """

        merged_text = "\n".join([section_path, doc.title, content[:500]])
        governance_text = " ".join(
            f"{key}:{doc.metadata.get(key) or ''}"
            for key in ("version", "effective_date", "expiry_date", "status", "version_status")
        )
        merged_text = "\n".join([merged_text, governance_text])
        return {
            "section_path": section_path or section_title,
            "section_level": str(section_level),
            "section_type": self._infer_section_type(section_title, content),
            "chunk_summary": self._build_chunk_summary(content),
            "contains_table": self._bool_text("|" in content or "\t" in content),
            "contains_steps": self._bool_text(bool(re.search(r"(^|\n)\s*(\d+\.|[-*])\s+", content))),
            "contains_contact": self._bool_text(bool(re.search(r"(联系人|电话|手机号|邮箱)", merged_text))),
            "contains_version_signal": self._bool_text(
                bool(
                    re.search(
                        r"(版本|生效日期|失效日期|作废|试行|现行|version:|effective_date:|expiry_date:|version_status:|status:)",
                        merged_text,
                    )
                )
            ),
            "contains_risk_signal": self._bool_text(
                bool(re.search(r"(风险|隐患|事故|应急|告警|异常)", merged_text))
            ),
            "topic_keywords": self._extract_topic_keywords(doc, section_path, content),
        }

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

    def _split_sections(self, text: str, *, profile: ChunkProfile | None = None) -> list[tuple[str, str, int, str]]:
        """按 Markdown 标题切分 section。

        当前切法偏工程实用：
        - 优先吃 `#` 标题
        - 保留 section path
        - 给后续 retrieval / citation 一条更稳定的章节路径

        为什么不是在这里重新理解 DOCX / HTML / PDF 原始结构：
        - 因为 parser 阶段已经先把不同格式规整成了统一文本
        - chunker 只消费统一表达，职责更单一，也更容易维护
        """

        if not text.strip():
            return []
        if profile and profile.split_pdf_by_page:
            page_sections = self._split_pdf_pages(text)
            if page_sections:
                return page_sections
        parts: list[tuple[str, str, int, str]] = []
        lines = text.split("\n")
        current_title = ""
        current_path = ""
        current_level = 1
        heading_stack: list[tuple[int, str]] = []
        buf: list[str] = []
        for line in lines:
            # 只识别 `#` 风格标题，这和 Markdown / HTML 转 Markdown 的输出形态相匹配。
            m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if m:
                if buf:
                    parts.append((current_title, current_path, current_level, "\n".join(buf).strip()))
                    buf = []
                current_title = m.group(2).strip()
                current_level = len(m.group(1))
                heading_stack = [(level, title) for level, title in heading_stack if level < current_level]
                heading_stack.append((current_level, current_title))
                current_path = " / ".join(title for _, title in heading_stack)
            else:
                buf.append(line)
        if buf:
            parts.append((current_title, current_path, current_level, "\n".join(buf).strip()))
        if not parts:
            return [("", "", 1, text)]
        return parts

    def _split_pdf_pages(self, text: str) -> list[tuple[str, str, int, str]]:
        """按 PDF 页码标记把文本切成页级 section。

        PDF 常见的问题是没有稳定标题层级，但页码又很重要。
        所以这里把“第 n 页”直接当作 section 标题，用来承接：
        - chunk.page
        - 引用里的页码信息
        - 后续按页定位 badcase
        """

        matches = list(_PAGE_MARKER.finditer(text))
        if not matches:
            return []
        parts: list[tuple[str, str, int, str]] = []
        for idx, match in enumerate(matches):
            page = match.group(1)
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            title = f"第{page}页"
            parts.append((title, title, 1, body))
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

    @staticmethod
    def _bool_text(value: bool) -> str:
        """把布尔值规整成稳定字符串。"""

        return "true" if value else "false"

    @staticmethod
    def _build_chunk_summary(content: str, max_chars: int = 96) -> str:
        """生成简短摘要，便于后续排序和调试。"""

        normalized = " ".join(content.split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 1].rstrip() + "…"

    @staticmethod
    def _infer_section_type(section_title: str, content: str) -> str:
        """根据标题和正文局部内容推断 section 类型。

        section_type 目前主要服务于：
        - retrieval boost
        - explainability
        - 前端引用展示
        """

        text = f"{section_title}\n{content[:200]}"
        rules = (
            (r"(适用范围|范围|总则)", "scope"),
            (r"(职责|责任|责任分工)", "responsibility"),
            (r"(流程|步骤|操作|处置|方案|实施)", "procedure"),
            (r"(例外|异常|特殊情况)", "exception"),
            (r"(附录|附件|附表)", "appendix"),
            (r"(定义|术语|说明)", "definition"),
        )
        for pattern, label in rules:
            if re.search(pattern, text):
                return label
        return "general"

    @staticmethod
    def _extract_topic_keywords(doc: Document, section_path: str, content: str) -> list[str]:
        """抽取少量稳定关键词，用于后续 metadata boost。

        这里不追求关键词算法复杂，而是追求：
        - 稳定
        - 成本低
        - 足够支撑 metadata boost 和 explainability
        """

        seeds: list[str] = []
        for key in (
            "business_domain",
            "process_stage",
            "equipment_type",
            "equipment_id",
            "system_name",
            "project_name",
            "doc_type",
        ):
            value = doc.metadata.get(key)
            if isinstance(value, str) and value.strip():
                seeds.append(value.strip())
        seeds.extend(_KEYWORD_RE.findall(section_path))
        seeds.extend(_KEYWORD_RE.findall(content[:160]))
        out: list[str] = []
        seen: set[str] = set()
        for item in seeds:
            token = item.strip()
            if len(token) < 2:
                continue
            normalized = token.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            out.append(token)
            if len(out) >= 8:
                break
        return out
