"""稀疏检索模块。

这里使用 BM25 做关键词召回。
它的优点是对精确术语、错误码、专有名词很敏感，
通常适合和向量检索一起做混合召回。
"""

from __future__ import annotations

import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from core.config.settings import Settings, get_settings
from core.models.document import TextChunk
from core.retrieval.schemas import RetrievedChunk


_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """把输入文本切成 BM25 可用的 token 列表。

    这里同时兼容英文单词、数字以及中文字符区间，
    目的是让错误码、路径片段和中英文混合问句都能较稳定地进入词法检索。
    """

    return [t.lower() for t in _TOKEN_RE.findall(text)]


class SparseRetriever:
    """基于 BM25 的稀疏检索器。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._corpus_tokens: list[list[str]] = []
        self._chunks: list[TextChunk] = []
        self._bm25: BM25Okapi | None = None

    def rebuild(self, chunks: Sequence[TextChunk]) -> None:
        """根据全部 chunk 重建 BM25 词项统计。"""

        self._chunks = list(chunks)
        self._corpus_tokens = [tokenize(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """执行 BM25 召回。"""

        k = top_k or self._settings.bm25_top_k
        if not self._bm25 or not self._chunks:
            return []
        q = tokenize(query)
        # `get_scores` 会给语料库中每个 chunk 一个相关性分数。
        scores = self._bm25.get_scores(q)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        out: list[RetrievedChunk] = []
        for idx, score in ranked:
            ch = self._chunks[idx]
            out.append(
                RetrievedChunk(
                    chunk_id=ch.metadata.chunk_id,
                    score=float(score),
                    content=ch.content,
                    metadata=ch.metadata,
                    trace={"retriever": "bm25"},
                )
            )
        return out
