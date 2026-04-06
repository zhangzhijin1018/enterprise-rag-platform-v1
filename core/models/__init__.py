"""core.models 包初始化模块。用于标记包边界，并为同目录下的实现提供统一导入入口。"""

from core.models.document import ChunkMetadata, Document, TextChunk

__all__ = ["ChunkMetadata", "Document", "TextChunk"]
