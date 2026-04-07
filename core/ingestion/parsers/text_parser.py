"""TXT 解析器模块。

`TXT` 看起来是最简单的文本格式，但在企业场景里其实非常常见：

- 导出的日志片段
- FAQ 草稿
- 规则说明
- 运维排障记录

这类文件通常已经是纯文本，所以这里不做复杂结构识别，
而是把重点放在：

1. 保留原始文本可检索性
2. 用统一的 `Document` 对象接入后续切块链路
3. 尽量兼容常见编码和换行差异
"""

from __future__ import annotations

from pathlib import Path

from core.ingestion.cleaners.text_cleaner import clean_text
from core.models.document import Document

from .base import BaseParser


class TextParser(BaseParser):
    """纯文本解析器。"""

    def parse(self, path: Path, source: str) -> Document:
        """读取 `.txt` 文件并转成统一 `Document`。

        这里使用 `errors="ignore"` 的原因很实际：

        - 企业里很多 TXT 文件并不总是严格 UTF-8
        - 与其因为个别非法字节直接报错，不如尽量保留可读内容继续入库

        如果后续要做更激进的编码识别，可以在这里再扩展 `chardet` 等方案。
        """

        raw = path.read_text(encoding="utf-8", errors="ignore")
        return Document(
            doc_id="",
            source=source,
            title=path.stem,
            content=clean_text(raw),
            mime_type="text/plain",
            metadata={},
        )
