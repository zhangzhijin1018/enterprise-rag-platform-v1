"""core.ingestion.parsers 包初始化模块。用于标记包边界，并为同目录下的实现提供统一导入入口。"""

from core.ingestion.parsers.base import BaseParser
from core.ingestion.parsers.csv_parser import CsvParser
from core.ingestion.parsers.docx_parser import DocxParser
from core.ingestion.parsers.html_parser import HtmlParser
from core.ingestion.parsers.markdown_parser import MarkdownParser
from core.ingestion.parsers.pdf_parser import PdfParser
from core.ingestion.parsers.pptx_parser import PptxParser
from core.ingestion.parsers.registry import get_parser_for_filename
from core.ingestion.parsers.text_parser import TextParser

__all__ = [
    "BaseParser",
    "CsvParser",
    "DocxParser",
    "HtmlParser",
    "MarkdownParser",
    "PdfParser",
    "PptxParser",
    "TextParser",
    "get_parser_for_filename",
]
