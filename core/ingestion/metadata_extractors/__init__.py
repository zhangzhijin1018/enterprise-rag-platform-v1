"""core.ingestion.metadata_extractors 包初始化模块。用于标记包边界，并为同目录下的实现提供统一导入入口。"""

from core.ingestion.metadata_extractors.basic import BasicMetadataExtractor

__all__ = ["BasicMetadataExtractor"]
