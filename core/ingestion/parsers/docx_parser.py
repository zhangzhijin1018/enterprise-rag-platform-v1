"""DOCX 解析器模块。负责从 Word 文档中提取连续文本并转成统一 Document 对象。"""

from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument

from core.ingestion.cleaners.text_cleaner import clean_text
from core.models.document import Document

from .base import BaseParser


class DocxParser(BaseParser):
    def parse(self, path: Path, source: str) -> Document:
        d = DocxDocument(str(path))
        lines: list[str] = []
        current_section = ""
        for p in d.paragraphs:
            text = (p.text or "").strip()
            if not text:
                continue
            style = (p.style.name if p.style else "") or ""
            if style.startswith("Heading"):
                current_section = text
                lines.append(f"## {text}")
            else:
                if current_section:
                    lines.append(f"[section:{current_section}] {text}")
                else:
                    lines.append(text)
        content = clean_text("\n".join(lines))
        return Document(
            doc_id="",
            source=source,
            title=path.stem,
            content=content,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            metadata={},
        )
