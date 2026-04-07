"""快速通道单元测试。"""

from __future__ import annotations

import pytest

from core.config.settings import Settings
from core.orchestration.fast_path import try_fast_path_answer
from core.retrieval.faq_retriever import FaqMatch
from core.retrieval.faq_store import FaqEntry


class _FakeCache:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], dict] = {}

    def get_json(self, prefix: str, payload: str):
        return self.values.get((prefix, payload))

    def set_json(self, prefix: str, payload: str, value, ttl_sec: int = 300) -> None:
        _ = ttl_sec
        self.values[(prefix, payload)] = value


class _FakeFaqRetriever:
    def __init__(self, matches: list[FaqMatch]) -> None:
        self.matches = matches
        self.calls = 0

    def search(self, query: str, top_k: int | None = None) -> list[FaqMatch]:
        _ = query, top_k
        self.calls += 1
        return self.matches


@pytest.mark.asyncio
async def test_fast_path_returns_mysql_faq_hit_and_caches_answer() -> None:
    settings = Settings(FAQ_BM25_THRESHOLD=0.85, ANSWER_CACHE_TTL_SEC=600)
    runtime = type("Runtime", (), {})()
    runtime.settings = settings
    runtime.cache = _FakeCache()
    runtime.faq_retriever = _FakeFaqRetriever(
        [
            FaqMatch(
                entry=FaqEntry(
                    entry_id=7,
                    question="错误码 E-1001 是什么？",
                    answer="表示 Redis connection failed。",
                    keywords="E-1001,Redis",
                    category="error_code",
                ),
                bm25_score=4.2,
                confidence=0.93,
            )
        ]
    )

    state = await try_fast_path_answer(runtime, "错误码 E-1001 是什么？")

    assert state is not None
    assert state["fast_path_source"] == "mysql_faq"
    assert "Redis connection failed" in state["answer"]
    assert runtime.faq_retriever.calls == 1
    cached = runtime.cache.get_json("answer", "错误码 e-1001 是什么？")
    assert cached is not None


@pytest.mark.asyncio
async def test_fast_path_prefers_redis_cache_before_mysql() -> None:
    settings = Settings(FAQ_BM25_THRESHOLD=0.85)
    cache = _FakeCache()
    cache.set_json(
        "answer",
        "milvus 和 zilliz cloud 有什么区别？",
        {
            "answer": "一个开源，一个托管。",
            "confidence": 0.98,
            "source_id": "faq:3",
            "title": "Milvus 和 Zilliz Cloud 有什么区别？",
            "section": "product_compare",
        },
    )
    runtime = type("Runtime", (), {})()
    runtime.settings = settings
    runtime.cache = cache
    runtime.faq_retriever = _FakeFaqRetriever([])

    state = await try_fast_path_answer(runtime, "Milvus 和 Zilliz Cloud 有什么区别？")

    assert state is not None
    assert state["fast_path_source"] == "redis_answer_cache"
    assert runtime.faq_retriever.calls == 0
