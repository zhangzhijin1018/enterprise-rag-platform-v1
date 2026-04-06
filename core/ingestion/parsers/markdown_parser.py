"""Markdown 解析器模块。用于读取 Markdown 文本并保留标题结构，方便后续语义切块。"""

from __future__ import annotations

from pathlib import Path

from core.ingestion.cleaners.text_cleaner import clean_text
from core.models.document import Document

from .base import BaseParser


class MarkdownParser(BaseParser):
    def parse(self, path: Path, source: str) -> Document:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return Document(
            doc_id="",
            source=source,
            title=path.stem,
            content=clean_text(raw),
            mime_type="text/markdown",
            metadata={},
        )
