"""core.orchestration.policies 包初始化模块。用于标记包边界，并为同目录下的实现提供统一导入入口。"""

from core.orchestration.policies.fallback import empty_retrieval_refusal

__all__ = ["empty_retrieval_refusal"]
