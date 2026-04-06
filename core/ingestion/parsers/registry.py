"""解析器注册表模块。根据文件名或扩展名选择正确的文档解析器。"""

from __future__ import annotations

from pathlib import Path

from core.ingestion.parsers.docx_parser import DocxParser
from core.ingestion.parsers.html_parser import HtmlParser
from core.ingestion.parsers.markdown_parser import MarkdownParser
from core.ingestion.parsers.pdf_parser import PdfParser

from .base import BaseParser

_SUFFIX_MAP: dict[str, BaseParser] = {
    ".pdf": PdfParser(),
    ".docx": DocxParser(),
    ".md": MarkdownParser(),
    ".markdown": MarkdownParser(),
    ".html": HtmlParser(),
    ".htm": HtmlParser(),
}


def get_parser_for_filename(filename: str) -> BaseParser:
    suf = Path(filename).suffix.lower()
    if suf not in _SUFFIX_MAP:
        raise ValueError(f"Unsupported file type: {suf}")
    return _SUFFIX_MAP[suf]
