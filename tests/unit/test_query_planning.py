"""查询规划与澄清判定单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.config.settings import get_settings
from core.generation.llm_client import LLMClient
from core.orchestration.nodes.clarify_query import clarify_query_node
from core.orchestration.query_expansion import build_query_plan
from core.retrieval.cache import RedisCache


def _runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.settings = get_settings()
    runtime.llm = LLMClient(runtime.settings)
    runtime.cache = RedisCache()
    return runtime


@pytest.mark.asyncio
async def test_clarify_node_flags_ambiguous_error_question() -> None:
    runtime = _runtime()
    state = {"question": "这个报错怎么解决？", "query_type": "general"}
    out = await clarify_query_node(state, runtime)
    assert out.get("need_clarify") is True
    assert "error_code" in (out.get("missing_slots") or [])
    assert "日志" in (out.get("clarify_question") or "")


@pytest.mark.asyncio
async def test_query_plan_decomposes_compare_question_without_llm() -> None:
    runtime = _runtime()
    plan = await build_query_plan(
        runtime,
        question="Milvus 和 Zilliz Cloud 有什么区别？",
        query_type="general",
    )
    assert "区别" in plan.rewritten_query
    assert len(plan.multi_queries) >= 2
    assert any("Milvus" in q for q in plan.keyword_queries)
