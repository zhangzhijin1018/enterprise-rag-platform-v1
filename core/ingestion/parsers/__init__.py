"""core.ingestion.parsers 包初始化模块。用于标记包边界，并为同目录下的实现提供统一导入入口。"""

from core.ingestion.parsers.base import BaseParser
from core.ingestion.parsers.docx_parser import DocxParser
from core.ingestion.parsers.html_parser import HtmlParser
from core.ingestion.parsers.markdown_parser import MarkdownParser
from core.ingestion.parsers.pdf_parser import PdfParser
from core.ingestion.parsers.registry import get_parser_for_filename

__all__ = [
    "BaseParser",
    "DocxParser",
    "HtmlParser",
    "MarkdownParser",
    "PdfParser",
    "get_parser_for_filename",
]
