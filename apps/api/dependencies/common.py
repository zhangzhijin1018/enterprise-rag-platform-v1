"""FastAPI 依赖注入辅助模块。

这里的目标是让所有路由都通过统一入口拿到 `RAGRuntime`，
而不是在路由里自己 new 检索器、模型或缓存。
"""

from core.services.runtime import RAGRuntime, get_runtime


def get_runtime_dep() -> RAGRuntime:
    """返回当前请求应复用的全局运行时。"""

    return get_runtime()
