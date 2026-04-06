"""集成测试夹具模块。负责准备 FastAPI 应用和集成测试共用上下文。"""

import pytest

from core.services.runtime import reset_runtime


@pytest.fixture(autouse=True)
def _isolate_runtime() -> None:
    reset_runtime()
    yield
    reset_runtime()
