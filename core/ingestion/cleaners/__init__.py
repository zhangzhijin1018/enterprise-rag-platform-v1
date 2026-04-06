"""core.ingestion.cleaners 包初始化模块。用于标记包边界，并为同目录下的实现提供统一导入入口。"""

from core.ingestion.cleaners.text_cleaner import clean_text

__all__ = ["clean_text"]
