"""FastAPI 依赖注入辅助模块。负责把全局运行时对象按请求注入到路由处理函数中。"""

from core.services.runtime import RAGRuntime, get_runtime


def get_runtime_dep() -> RAGRuntime:
    return get_runtime()
