"""文档解析器抽象基类模块。

parser 层在 ingestion 链路中的职责很明确：

1. 读取不同格式的原始文件
2. 尽量保留原文里的结构信息，例如标题、页码、表格、列表
3. 输出统一的 :class:`Document`，供后续 metadata 抽取、切块、入库复用

它不负责：

- 生成最终 `doc_id`
- 做 embedding
- 决定检索策略

这样分层的好处是，后续新增文件类型时，只需要关心“如何把该格式稳定转成统一文本”，
而不用在 parser 里耦合更多下游逻辑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.models.document import Document


class BaseParser(ABC):
    """所有文件解析器都必须遵守的最小契约。

    统一约束 `parse(path, source) -> Document` 有两个直接收益：

    1. `registry` 可以按文件后缀动态路由 parser，而不关心具体实现细节
    2. ingestion pipeline 可以把 DOCX / PDF / PPTX / CSV 一视同仁地继续处理
    """

    @abstractmethod
    def parse(self, path: Path, source: str) -> Document:
        """把原始文件解析成统一 `Document`。

        Args:
            path: 本地文件路径，由上游上传或批量入库流程传入。
            source: 来源描述，一般用于保留原始文件名、对象存储路径或外部系统来源。

        Returns:
            统一的 `Document` 对象。此时内容已经是“适合继续切块和检索”的标准文本，
            但文档唯一标识、更多企业 metadata 往往还会在后续阶段继续补齐。
        """

        raise NotImplementedError
