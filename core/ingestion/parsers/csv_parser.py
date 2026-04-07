"""CSV 解析器模块。

企业知识库里经常会出现表格型资料，例如：

- FAQ 导出
- 错误码对照表
- 配置项说明
- 工单字段快照

这类数据如果直接按原始 CSV 文本整段入库，检索效果通常并不好，
因为逗号分隔的扁平文本缺少足够稳定的语义边界。

所以这里采用一个更偏 RAG 友好的策略：

1. 读出表头
2. 每一行转成“字段名: 字段值”的结构化文本块
3. 用 Markdown 标题标记行号，方便后续切块与引用追踪
"""

from __future__ import annotations

import csv
from pathlib import Path

from core.ingestion.cleaners.text_cleaner import clean_text
from core.models.document import Document

from .base import BaseParser


class CsvParser(BaseParser):
    """CSV / TSV 风格表格解析器。"""

    def __init__(self, *, delimiter: str = ",") -> None:
        self.delimiter = delimiter

    def parse(self, path: Path, source: str) -> Document:
        """把 CSV 文件转成适合检索的结构化文本。

        当前算法：

        - 首行作为 header
        - 后续每一行输出为一个小节
        - 每列转成 `列名: 值`

        示例：

        ```text
        ## Row 1
        code: E-1001
        reason: Redis connection failed
        solution: Check network and password
        ```

        这样做的好处：

        - 比直接拼逗号分隔文本更利于 BM25 和 dense retrieval
        - 字段名会成为稳定检索锚点
        - 行号也方便后续引用定位
        """

        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            rows = list(reader)

        if not rows:
            content = ""
            metadata = {"rows": 0, "columns": 0}
        else:
            headers = [clean_text(cell).strip() or f"column_{idx + 1}" for idx, cell in enumerate(rows[0])]
            lines: list[str] = []
            for row_idx, row in enumerate(rows[1:], start=1):
                if not any((cell or "").strip() for cell in row):
                    continue
                lines.append(f"## Row {row_idx}")
                for col_idx, value in enumerate(row):
                    header = headers[col_idx] if col_idx < len(headers) else f"column_{col_idx + 1}"
                    text = clean_text(value).strip()
                    if text:
                        lines.append(f"{header}: {text}")
            # 如果文件只有表头没有数据，仍然把表头保留下来，避免完全空文档。
            if not lines and headers:
                lines.append("## Header")
                lines.extend(headers)
            content = clean_text("\n".join(lines))
            metadata = {"rows": max(0, len(rows) - 1), "columns": len(headers)}

        return Document(
            doc_id="",
            source=source,
            title=path.stem,
            content=content,
            mime_type="text/csv",
            metadata=metadata,
        )
