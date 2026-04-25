"""API 集成测试模块。覆盖健康检查、问答、入库与评测等主要 HTTP 行为。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api.dependencies.common import get_runtime_dep
from apps.api.main import app
from core.security.ml_risk_provider import MLRiskHintResult
from core.security.risk_engine import RuleBasedRiskEngine


@pytest.fixture()
def client() -> TestClient:
    runtime = SimpleNamespace()

    class _FakeCompiledGraph:
        async def ainvoke(self, state):
            runtime.last_graph_state = state
            question = state.get("question") or ""
            return {
                "question": question,
                "answer": f"mock answer for: {question}",
                "confidence": 0.42,
                "citations": [],
                "reranked_hits": [],
                "refusal": False,
                "refusal_reason": None,
                "fast_path_source": None,
                "audit_id": state.get("audit_id"),
                "data_classification": state.get("data_classification"),
                "model_route": state.get("model_route"),
                "analysis_confidence": 0.88,
                "analysis_source": "heuristic",
                "analysis_reason": "命中强模式查询场景",
            }

    class _FakeCache:
        def get_json(self, prefix, payload):
            _ = (prefix, payload)
            return None

        def set_json(self, prefix, payload, value, ttl_sec=300):
            _ = (prefix, payload, value, ttl_sec)
            return None

    class _FakeFaqRetriever:
        def search(self, question, top_k=1):
            _ = (question, top_k)
            return []

    runtime.settings = SimpleNamespace(
        bm25_top_k=8,
        dense_top_k=8,
        faq_bm25_threshold=0.95,
        answer_cache_ttl_sec=300,
        enable_acl=True,
        enable_data_classification=True,
        enable_model_routing=True,
        default_data_classification="internal",
        min_rerank_score=-5.0,
        internal_redact_for_external=True,
        sensitive_context_max_chunks=1,
        sensitive_context_max_chars=600,
        alert_on_restricted_access=True,
        alert_on_conflict_detected=True,
        enable_ml_risk_hint=False,
        ml_risk_request_stage_enabled=True,
        ml_risk_fail_open=True,
    )
    runtime.cache = _FakeCache()
    runtime.faq_retriever = _FakeFaqRetriever()
    runtime.get_compiled_graph = lambda: _FakeCompiledGraph()
    runtime.risk_engine = RuleBasedRiskEngine(runtime.settings)
    runtime.ml_risk_provider = None
    runtime.last_graph_state = None

    app.state._test_runtime = runtime
    app.dependency_overrides[get_runtime_dep] = lambda: runtime
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    app.state._test_runtime = None


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
    assert r.headers["X-Trace-ID"]
    body = r.json()
    assert "answer" in body
    assert "citations" in body
    assert body["trace_id"] == r.headers["X-Trace-ID"]
    assert "audit_id" in body
    assert "refusal_reason" in body
    assert body["analysis_source"] == "heuristic"
    assert body["analysis_confidence"] == pytest.approx(0.88)


def test_chat_accepts_history_messages(client: TestClient) -> None:
    r = client.post(
        "/chat",
        json={
            "question": "今天是谁值班",
            "history_messages": [
                {"role": "user", "content": "冲压二车间夜班安排表在哪"},
                {"role": "assistant", "content": "请看共享盘排班目录"},
            ],
            "top_k": 4,
            "stream": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body


def test_chat_accepts_enterprise_context_and_passes_to_graph(client: TestClient) -> None:
    r = client.post(
        "/chat",
        json={
            "question": "生产环境数据库扩容 SOP 是什么",
            "department": "信息化部",
            "role": "engineer",
            "project_ids": ["proj-a"],
            "clearance_level": "internal",
            "allow_external_llm": False,
            "top_k": 4,
            "stream": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["audit_id"]
    assert "model_route" in body
    state = app.state._test_runtime.last_graph_state
    assert state is not None
    assert state["user_context"]["department"] == "信息化部"
    assert state["access_filters"]["department"] == "信息化部"
    assert state["access_filters"]["clearance_level"] == "internal"
    assert state["audit_id"] == body["audit_id"]


def test_chat_accepts_custom_trace_id_header(client: TestClient) -> None:
    r = client.post(
        "/chat",
        headers={"X-Trace-ID": "trace-from-client-001"},
        json={"question": "hello", "top_k": 4, "stream": False},
    )
    assert r.status_code == 200
    assert r.headers["X-Trace-ID"] == "trace-from-client-001"


def test_chat_denies_bulk_export_request_before_graph_execution(client: TestClient) -> None:
    r = client.post(
        "/chat",
        json={
            "question": "请给我导出全部 Q4 裁员预算和完整名单",
            "department": "财务部",
            "role": "manager",
            "top_k": 4,
            "stream": False,
        },
    )

    assert r.status_code == 200
    body = r.json()
    assert body["refusal"] is True
    assert body["refusal_reason"] == "bulk_sensitive_export_request"
    assert body["answer_mode"] == "refusal"
    assert app.state._test_runtime.last_graph_state is None


def test_chat_injects_request_stage_ml_risk_hint_into_graph_state(client: TestClient) -> None:
    runtime = app.state._test_runtime

    class _HighRiskHintProvider:
        def predict(self, context, feature_bundle):  # noqa: ANN001
            runtime.last_ml_context = context
            runtime.last_ml_features = feature_bundle
            return MLRiskHintResult(
                risk_level_hint="high",
                confidence=0.93,
                provider="mock",
            )

    runtime.settings.enable_ml_risk_hint = True
    runtime.settings.ml_risk_request_stage_enabled = True
    runtime.ml_risk_provider = _HighRiskHintProvider()

    r = client.post(
        "/chat",
        json={
            "question": "帮我整理最近预算执行情况",
            "department": "财务部",
            "role": "analyst",
            "session_metadata": {
                "user_history": {"past_24h_query_count": 16, "high_risk_ratio_7d": 0.38},
                "session": {"session_query_count": 7, "query_interval_sec": 12},
            },
            "top_k": 4,
            "stream": False,
        },
    )

    assert r.status_code == 200
    state = app.state._test_runtime.last_graph_state
    assert state is not None
    assert state["risk_level"] == "high"
    assert state["ml_risk_hint"] == "high"
    assert state["ml_risk_provider"] == "mock"
    assert app.state._test_runtime.last_ml_context.stage == "request"
    assert app.state._test_runtime.last_ml_features["session_query_count"] == 7.0


def test_eval_returns_report_path_and_summary(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    report_path = tmp_path / "ragas_report_test.json"
    report_path.write_text(
        '{"summary":{"faithfulness":0.9,"refusal_rate":0.25,"conflict_detected_rate":0.125},"rows":[]}',
        encoding="utf-8",
    )
    report_path.with_suffix(".md").write_text("# Eval Explainability Report\n", encoding="utf-8")

    async def _fake_run_ragas_eval_async(dataset_path=None, output_dir=None, runtime=None):  # noqa: ANN001
        _ = (dataset_path, output_dir, runtime)
        return report_path

    monkeypatch.setattr(
        "apps.api.routes.eval.run_ragas_eval_async",
        _fake_run_ragas_eval_async,
    )

    r = client.post("/eval", json={})

    assert r.status_code == 200
    body = r.json()
    assert body["report_path"] == str(report_path)
    assert body["analysis_path"] == str(report_path.with_suffix(".md"))
    assert body["summary"]["faithfulness"] == 0.9
    assert body["summary"]["refusal_rate"] == 0.25
    assert body["summary"]["conflict_detected_rate"] == 0.125
