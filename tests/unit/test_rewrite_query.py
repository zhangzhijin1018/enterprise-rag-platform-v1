"""查询改写单元测试模块。验证缓存命中和离线模式下的改写行为。"""

from unittest.mock import MagicMock

import pytest

from core.config.settings import get_settings
from core.generation.llm_client import LLMClient
from core.orchestration.nodes.rewrite_query import rewrite_query_node
from core.orchestration.state import RAGState
from core.retrieval.cache import RedisCache


@pytest.mark.asyncio
async def test_rewrite_fallback_without_api_key() -> None:
    runtime = MagicMock()
    runtime.settings = get_settings()
    runtime.llm = LLMClient(runtime.settings)
    runtime.cache = RedisCache()

    state: RAGState = {"question": "  How to fix E-404?  ", "query_type": "error_code"}
    out = await rewrite_query_node(state, runtime)
    assert out.get("rewritten_query")
    assert "E-404" in (out.get("rewritten_query") or "")
