"""API 集成测试模块。覆盖健康检查、问答、入库与评测等主要 HTTP 行为。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"erp_" in r.content or b"python" in r.content


def test_chat_empty_index(client: TestClient) -> None:
    r = client.post("/chat", json={"question": "hello", "top_k": 4, "stream": False})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "citations" in body
