"""PDF 解析器模块。

PDF 和 DOCX/PPTX 最大的区别在于：

- 它通常缺少可直接复用的“标题样式”语义
- 版面信息比样式信息更重要
- 真实企业文档里经常需要保留页码，方便引用和审计

所以这里的核心策略不是推断复杂结构，而是稳妥地做两件事：

1. 逐页抽取文本
2. 在文本里显式写入页码锚点，供后续切块和引用链路继续继承
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from core.ingestion.cleaners.text_cleaner import clean_text
from core.models.document import Document

from .base import BaseParser


class PdfParser(BaseParser):
    """把 PDF 转成带页码锚点的统一 `Document`。"""

    def parse(self, path: Path, source: str) -> Document:
        """逐页抽取 PDF 文本。

        PDF 的解析稳定性通常受原文件质量影响较大，因此这里优先追求：

        - 行为稳定
        - 输出结构简单
        - 页码可追踪
        """

        reader = PdfReader(str(path))
        parts: list[str] = []
        for i, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            block = clean_text(raw)
            if block:
                # 页码注释会在后续切块阶段被继承，方便引用“来自第几页”。
                parts.append(f"<!-- page:{i} -->\n{block}")
        content = clean_text("\n\n".join(parts))
        return Document(
            # parser 不负责生成持久化主键。
            doc_id="",
            source=source,
            title=path.stem,
            content=content,
            mime_type="application/pdf",
            # PDF 最关键的文档级结构信息就是页数，先保留在 metadata 中。
            metadata={"pages": len(reader.pages)},
        )
