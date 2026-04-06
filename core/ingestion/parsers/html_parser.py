"""HTML 解析器模块。负责移除网页噪声并把主要内容规范化为纯文本。"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from core.ingestion.cleaners.text_cleaner import clean_text
from core.models.document import Document

from .base import BaseParser


def _heading_level(tag_name: str) -> int | None:
    if tag_name.startswith("h") and len(tag_name) == 2 and tag_name[1].isdigit():
        return int(tag_name[1])
    return None


class HtmlParser(BaseParser):
    def parse(self, path: Path, source: str) -> Document:
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        lines: list[str] = []
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else path.stem
        body = soup.body or soup
        for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            t = el.get_text(" ", strip=True)
            if not t:
                continue
            lvl = _heading_level(el.name)
            if lvl is not None:
                lines.append(f"{'#' * lvl} {t}")
            else:
                lines.append(t)
        content = clean_text("\n".join(lines))
        return Document(
            doc_id="",
            source=source,
            title=title or path.stem,
            content=content,
            mime_type="text/html",
            metadata={},
        )
