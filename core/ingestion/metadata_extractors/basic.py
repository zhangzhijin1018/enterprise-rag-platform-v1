"""基础元数据提取模块。用于补全文档 ID、标题等后续检索和引用需要的字段。"""

from __future__ import annotations

import uuid
from pathlib import Path

from core.models.document import Document


class BasicMetadataExtractor:
    def ensure_doc_id(self, doc: Document) -> Document:
        if doc.doc_id:
            return doc
        return doc.model_copy(update={"doc_id": str(uuid.uuid4())})

    def infer_title_from_filename(self, path: str | Path, doc: Document) -> Document:
        if doc.title:
            return doc
        name = Path(path).stem
        return doc.model_copy(update={"title": name})
