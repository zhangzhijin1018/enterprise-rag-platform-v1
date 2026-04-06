"""PDF 解析器模块。负责逐页抽取文本，并写入页码标记供切块阶段继承。"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from core.ingestion.cleaners.text_cleaner import clean_text
from core.models.document import Document

from .base import BaseParser


class PdfParser(BaseParser):
    def parse(self, path: Path, source: str) -> Document:
        reader = PdfReader(str(path))
        parts: list[str] = []
        for i, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            block = clean_text(raw)
            if block:
                parts.append(f"<!-- page:{i} -->\n{block}")
        content = clean_text("\n\n".join(parts))
        return Document(
            doc_id="",
            source=source,
            title=path.stem,
            content=content,
            mime_type="application/pdf",
            metadata={"pages": len(reader.pages)},
        )
