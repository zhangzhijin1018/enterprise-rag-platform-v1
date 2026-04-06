"""测试全局夹具模块。为不同测试目录提供共享初始化逻辑与公共 fixture。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def test_env(tmp_path_factory: pytest.TempPathFactory) -> None:
    root = tmp_path_factory.mktemp("vector")
    os.environ.setdefault("VECTOR_STORE_PATH", str(root))
    os.environ.setdefault("REDIS_URL", "")
    os.environ.setdefault("OPENAI_API_KEY", "")
