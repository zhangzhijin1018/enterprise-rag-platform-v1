"""企业访问控制基础模块测试。"""

from __future__ import annotations

from types import SimpleNamespace

from apps.api.schemas.chat import ChatRequest, ChatResponse
from core.config.settings import Settings
from core.models.document import ChunkMetadata
from core.retrieval.access_control import (
    build_access_filters,
    build_retrieval_acl_filters,
    is_chunk_accessible,
    resolve_data_classification,
    resolve_model_route,
)


def _settings(**overrides):
    base = {
        "enable_acl": True,
        "enable_data_classification": True,
        "enable_model_routing": True,
        "default_data_classification": "internal",
        "allow_external_llm_for_sensitive": False,
        "local_only_classifications": ["restricted"],
        "acl_strict_mode": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _metadata(**extra) -> ChunkMetadata:
    return ChunkMetadata(
        doc_id="doc-1",
        chunk_id="chunk-1",
        source="mock.md",
        title="mock",
        extra=extra,
    )


def test_chat_request_supports_enterprise_fields_with_defaults() -> None:
    req = ChatRequest(question="报销流程是什么？")
    assert req.require_citations is True
    assert req.project_ids == []
    assert req.session_metadata == {}
    assert req.allow_external_llm is None


def test_chat_response_exposes_security_fields() -> None:
    resp = ChatResponse(
        answer="mock answer",
        confidence=0.9,
        citations=[],
        retrieved_chunks=[],
    )
    assert resp.refusal is False
    assert resp.refusal_reason is None
    assert resp.model_route is None
    assert resp.audit_id is None


def test_settings_support_security_defaults_and_csv_env(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_ONLY_CLASSIFICATIONS", "restricted,sensitive")
    monkeypatch.setenv("SUPPORTED_DEPARTMENTS", "生产运营部,信息化部")
    settings = Settings(_env_file=None)
    assert settings.project_name == "Xinjiang Energy Knowledge Copilot"
    assert settings.local_only_classifications == ["restricted", "sensitive"]
    assert settings.supported_departments == ["生产运营部", "信息化部"]


def test_build_access_filters_from_user_context() -> None:
    filters = build_access_filters(
        {
            "user_id": "u-1",
            "department": "生产运营部",
            "role": "engineer",
            "project_ids": ["p-1", "p-2"],
            "clearance_level": "sensitive",
        },
        _settings(),
    )
    assert filters == {
        "user_id": "u-1",
        "department": "生产运营部",
        "role": "engineer",
        "project_ids": ["p-1", "p-2"],
        "clearance_level": "sensitive",
    }


def test_build_retrieval_acl_filters_maps_user_context_to_metadata_keys() -> None:
    filters = build_retrieval_acl_filters(
        {
            "user_id": "u-1",
            "department": "生产运营部",
            "role": "engineer",
            "project_ids": ["p-1", "p-2"],
            "clearance_level": "sensitive",
        }
    )
    assert filters == {
        "allowed_users": "u-1",
        "allowed_departments": "生产运营部",
        "allowed_roles": "engineer",
        "project_ids": ["p-1", "p-2"],
    }


def test_is_chunk_accessible_allows_matching_department_and_role() -> None:
    metadata = _metadata(
        data_classification="internal",
        allowed_departments=["生产运营部"],
        allowed_roles=["engineer"],
        project_ids=["p-1"],
    )
    assert is_chunk_accessible(
        metadata,
        {
            "department": "生产运营部",
            "role": "engineer",
            "project_ids": ["p-1"],
            "clearance_level": "internal",
        },
        _settings(),
    )


def test_is_chunk_accessible_rejects_unauthorized_role() -> None:
    metadata = _metadata(
        data_classification="internal",
        allowed_roles=["manager"],
    )
    assert not is_chunk_accessible(
        metadata,
        {
            "department": "生产运营部",
            "role": "engineer",
            "project_ids": ["p-1"],
            "clearance_level": "internal",
        },
        _settings(),
    )


def test_is_chunk_accessible_rejects_sensitive_chunk_without_acl_in_strict_mode() -> None:
    metadata = _metadata(data_classification="sensitive")
    assert not is_chunk_accessible(
        metadata,
        {
            "department": "生产运营部",
            "role": "engineer",
            "project_ids": ["p-1"],
            "clearance_level": "sensitive",
        },
        _settings(),
    )


def test_resolve_model_route_for_sensitive_and_restricted_data() -> None:
    settings = _settings()
    assert resolve_model_route(settings=settings, data_classification="restricted") == "local_only"
    assert resolve_model_route(settings=settings, data_classification="sensitive") == "local_preferred"
    assert (
        resolve_model_route(
            settings=settings,
            data_classification="sensitive",
            allow_external_llm=True,
        )
        == "external_allowed"
    )


def test_resolve_data_classification_returns_highest_level() -> None:
    result = resolve_data_classification(
        [
            _metadata(data_classification="internal"),
            _metadata(data_classification="restricted"),
            _metadata(data_classification="public"),
        ]
    )
    assert result == "restricted"
