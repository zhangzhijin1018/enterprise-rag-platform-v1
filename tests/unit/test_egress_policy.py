"""数据分级出域控制单元测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.generation.egress_policy import prepare_contexts_for_generation, redact_text_for_external
from core.models.document import ChunkMetadata
from core.orchestration.nodes.generate_answer import generate_answer_node
from core.retrieval.schemas import RetrievedChunk


def _settings(**overrides):
    base = {
        "default_data_classification": "internal",
        "local_only_classifications": ["restricted"],
        "internal_redact_for_external": True,
        "enable_local_fallback_generation": True,
        "sensitive_context_max_chunks": 1,
        "sensitive_context_max_chars": 40,
        "min_rerank_score": -5.0,
        "audit_log_enabled": False,
        "audit_log_preview_chars": 120,
        "audit_log_redact_content": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _hit(chunk_id: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=0.9,
        content=content,
        metadata=ChunkMetadata(
            doc_id=f"doc-{chunk_id}",
            chunk_id=chunk_id,
            source="mock.md",
            title="mock",
        ),
        trace={},
    )


def test_redact_text_for_external_masks_common_sensitive_patterns() -> None:
    text = "联系人张三，邮箱 test@example.com，手机号 13812345678，身份证 110101199001011234"
    redacted = redact_text_for_external(text)
    assert "test@example.com" not in redacted
    assert "13812345678" not in redacted
    assert "110101199001011234" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_ID]" in redacted


def test_prepare_contexts_for_generation_uses_minimal_redacted_context_for_sensitive() -> None:
    contexts = [
        _hit("c1", "手机号 13812345678，扩容窗口为今晚 22:00。"),
        _hit("c2", "第二条内容不应继续出域。"),
    ]
    prepared, decision = prepare_contexts_for_generation(
        contexts,
        settings=_settings(),
        data_classification="sensitive",
        model_route="local_preferred",
    )

    assert decision["allowed"] is True
    assert decision["strategy"] == "minimal_redacted_context"
    assert len(prepared) == 1
    assert "13812345678" not in prepared[0].content
    assert prepared[0].trace["egress_redacted"] is True


def test_prepare_contexts_for_generation_blocks_restricted_data() -> None:
    prepared, decision = prepare_contexts_for_generation(
        [_hit("c1", "restricted content")],
        settings=_settings(),
        data_classification="restricted",
        model_route="local_only",
    )
    assert prepared == []
    assert decision["allowed"] is False
    assert decision["refusal_reason"] == "restricted_data_local_only"


@pytest.mark.asyncio
async def test_generate_answer_node_uses_local_fallback_when_model_route_is_local_only() -> None:
    class _FakeLLM:
        async def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("restricted 数据不应该调用外部 LLM")

    runtime = SimpleNamespace(settings=_settings(), llm=_FakeLLM())
    state = {
        "question": "预算方案是什么",
        "reranked_hits": [
            _hit("c1", "restricted content").model_dump(mode="json"),
        ],
        "data_classification": "restricted",
        "model_route": "local_only",
        "audit_id": "audit-1",
        "user_context": {"department": "财务部"},
        "answer_mode": "grounded_answer",
    }

    out = await generate_answer_node(state, runtime)

    assert out["refusal"] is False
    assert out["answer_mode"] == "local_grounded_answer"
    assert out["citations"]
    assert "本地受限上下文" in out["answer"]
