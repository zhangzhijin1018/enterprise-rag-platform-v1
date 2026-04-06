"""文档解析器抽象基类模块。约束不同文件类型解析器的统一输入输出接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.models.document import Document


class BaseParser(ABC):
    @abstractmethod
    def parse(self, path: Path, source: str) -> Document:
        raise NotImplementedError
