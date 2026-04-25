"""日志上下文与审计链路测试。"""

from __future__ import annotations

from types import SimpleNamespace

from core.observability import clear_request_log_context, set_request_log_context
from core.observability.audit import build_audit_event


def test_build_audit_event_includes_trace_id_from_request_context() -> None:
    settings = SimpleNamespace(
        audit_log_preview_chars=120,
        audit_log_redact_content=True,
    )
    set_request_log_context(trace_id="trace-001", user_id="u1")
    try:
        event = build_audit_event(
            stage="request_received",
            audit_id="audit-001",
            question="生产环境数据库扩容 SOP 是什么",
            user_context={"user_id": "u1", "department": "信息化部", "role": "engineer"},
            settings=settings,
            state={},
        )
    finally:
        clear_request_log_context()

    assert event["trace_id"] == "trace-001"
    assert event["audit_id"] == "audit-001"
    assert event["user_id"] == "u1"
