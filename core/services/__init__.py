"""core.services 包初始化模块。用于标记包边界，并为同目录下的实现提供统一导入入口。"""

from core.services.runtime import RAGRuntime, get_runtime

__all__ = ["RAGRuntime", "get_runtime"]
