"""apps.api.dependencies 包初始化模块。用于标记包边界，并为同目录下的实现提供统一导入入口。"""

from apps.api.dependencies.common import get_runtime_dep

__all__ = ["get_runtime_dep"]
