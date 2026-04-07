"""解析器注册表模块。根据文件名或扩展名选择正确的文档解析器。"""

from __future__ import annotations

from pathlib import Path

from core.ingestion.parsers.csv_parser import CsvParser
from core.ingestion.parsers.docx_parser import DocxParser
from core.ingestion.parsers.html_parser import HtmlParser
from core.ingestion.parsers.markdown_parser import MarkdownParser
from core.ingestion.parsers.pdf_parser import PdfParser
from core.ingestion.parsers.pptx_parser import PptxParser
from core.ingestion.parsers.text_parser import TextParser

from .base import BaseParser

_SUFFIX_MAP: dict[str, BaseParser] = {
    ".csv": CsvParser(),
    ".pdf": PdfParser(),
    ".docx": DocxParser(),
    ".pptx": PptxParser(),
    ".md": MarkdownParser(),
    ".markdown": MarkdownParser(),
    ".txt": TextParser(),
    ".html": HtmlParser(),
    ".htm": HtmlParser(),
}


def get_parser_for_filename(filename: str) -> BaseParser:
    """根据文件名后缀返回解析器实例。

    这里故意保持实现简单直接，而不是做复杂插件发现：

    1. 入库链路是高频核心路径，显式映射最稳定；
    2. 新增格式时改动点非常集中，便于排查；
    3. 测试时也更容易明确断言“某个扩展名到底会落到哪个 parser”。
    """

    suf = Path(filename).suffix.lower()
    if suf not in _SUFFIX_MAP:
        raise ValueError(f"Unsupported file type: {suf}")
    return _SUFFIX_MAP[suf]
