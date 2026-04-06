"""core.generation.prompts 包初始化模块。用于标记包边界，并为同目录下的实现提供统一导入入口。"""

from core.generation.prompts.templates import (
    GROUNDED_ANSWER_SYSTEM,
    QUERY_REWRITE_SYSTEM,
)

__all__ = ["GROUNDED_ANSWER_SYSTEM", "QUERY_REWRITE_SYSTEM"]
