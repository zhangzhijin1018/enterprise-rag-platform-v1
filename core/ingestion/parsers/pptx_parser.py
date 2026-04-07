"""PPTX 解析器模块。

PPT / PPTX 在企业知识库里也非常常见：

- 产品宣讲材料
- 培训课件
- 方案评审稿
- 架构分享文档

这类文档的难点在于：

- 内容被切散在多个 slide
- 标题和正文的层级很重要
- 表达往往是短句和 bullet points，而不是完整段落

因此这里的策略是：

1. 逐页读取 slide
2. 优先提取标题
3. 再按 shape 顺序提取文本
4. 输出为带 slide 标记和标题的结构化文本
"""

from __future__ import annotations

from pathlib import Path

from core.ingestion.cleaners.text_cleaner import clean_text
from core.models.document import Document

from .base import BaseParser


def _safe_slide_title(slide) -> str:
    """尽量从 slide 中提取标题。"""

    title_shape = getattr(slide.shapes, "title", None)
    if title_shape is not None:
        text = clean_text(getattr(title_shape, "text", "")).strip()
        if text:
            return text
    return ""


class PptxParser(BaseParser):
    """PPTX 解析器。"""

    def parse(self, path: Path, source: str) -> Document:
        """把 PPTX 文档转成结构化文本。

        这里把 `pptx` 的导入放到函数内部，而不是模块顶层，
        是为了让项目在未安装 `python-pptx` 时仍能正常启动其它链路。

        只有真正入库 `.pptx` 文件时，才会显式要求这个依赖。
        """

        try:
            from pptx import Presentation
        except ModuleNotFoundError as exc:  # pragma: no cover - 依赖缺失属于环境问题，不是逻辑分支
            raise RuntimeError(
                "PPTX parsing requires `python-pptx`. Install it before ingesting .pptx files."
            ) from exc

        presentation = Presentation(str(path))
        lines: list[str] = []
        first_title = ""
        for slide_idx, slide in enumerate(presentation.slides, start=1):
            slide_title = _safe_slide_title(slide)
            if slide_title and not first_title:
                first_title = slide_title

            # 每个 slide 都显式写出一个标题块，便于后续切块时保留页内结构。
            if slide_title:
                lines.append(f"# Slide {slide_idx}: {slide_title}")
            else:
                lines.append(f"# Slide {slide_idx}")

            for shape in slide.shapes:
                # 标题已经单独处理过，避免重复写入。
                if getattr(slide.shapes, "title", None) is shape:
                    continue
                text = clean_text(getattr(shape, "text", "")).strip()
                if text:
                    lines.append(text)

        content = clean_text("\n".join(lines))
        return Document(
            doc_id="",
            source=source,
            title=first_title or path.stem,
            content=content,
            mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            metadata={"slides": len(presentation.slides)},
        )
