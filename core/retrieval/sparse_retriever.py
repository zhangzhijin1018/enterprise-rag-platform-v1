"""稀疏检索模块。

这里使用 BM25 做关键词召回。
它的优点是对精确术语、错误码、专有名词很敏感，
通常适合和向量检索一起做混合召回。
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from core.config.settings import Settings, get_settings
from core.models.document import TextChunk
from core.observability import get_logger
from core.retrieval.bgem3_backend import BGEM3Backend
from core.retrieval.metadata_filters import chunk_matches_filters
from core.retrieval.milvus_retriever import MilvusDenseRetriever
from core.retrieval.schemas import RetrievedChunk

logger = get_logger(__name__)

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
        self._bgem3 = BGEM3Backend(self._settings)
        self._milvus: MilvusDenseRetriever | None = None
        self._corpus_tokens: list[list[str]] = []
        self._chunks: list[TextChunk] = []
        self._bm25: BM25Okapi | None = None
        self._sparse_matrix: Any | None = None

    def _get_milvus(self) -> MilvusDenseRetriever:
        """懒加载 Milvus 检索后端，避免启动时重复建立连接。"""

        if self._milvus is None:
            self._milvus = MilvusDenseRetriever(self._settings)
        return self._milvus

    def rebuild(self, chunks: Sequence[TextChunk]) -> None:
        """根据全部 chunk 重建 BM25 词项统计。

        第三轮增强后，索引里会同时存在 parent / child 两层 chunk。
        这里刻意只把 `searchable=True` 的 chunk 放进 BM25，
        也就是默认只索引 child chunks。

        原因：
        - child chunk 更短，更适合精准召回；
        - parent chunk 更长，更适合生成时提供完整上下文；
        - 如果把 parent 和 child 都塞进 BM25，检索结果会出现大量“长块压制短块”的噪声。
        """

        self._chunks = [c for c in chunks if c.searchable]
        self._bm25 = None
        self._sparse_matrix = None
        self._corpus_tokens = []
        if not self._chunks:
            return
        if (
            getattr(self._settings, "vector_backend", "") == "milvus"
            and self._bgem3.enabled
            and self._bgem3.get_function() is not None
        ):
            # 当前项目已经把 BGEM3 sparse 向量落到 Milvus。
            # 因此 Milvus 路线下，稀疏检索不再在内存里重建 sparse_matrix，
            # 只保留 searchable chunks 视图供本地 filter / fallback 使用。
            return
        if self._bgem3.enabled and self._bgem3.get_function() is not None:
            self._sparse_matrix = self._bgem3.encode_documents([c.content for c in self._chunks])["sparse"]
            return
        self._corpus_tokens = [tokenize(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def _search_with_bgem3_sparse(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """使用 BGEM3 产生的 sparse 向量执行检索。"""

        if self._sparse_matrix is None:
            return []
        query_sparse = self._bgem3.encode_queries([query])["sparse"]
        product = self._sparse_matrix @ query_sparse.T
        if hasattr(product, "toarray"):
            product = product.toarray()
        scores = np.asarray(product, dtype=np.float32).reshape(-1)
        ranked = sorted(
            (
                (idx, score)
                for idx, score in enumerate(scores)
                if chunk_matches_filters(self._chunks[idx].metadata, filters)
            ),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:top_k]
        out: list[RetrievedChunk] = []
        for idx, score in ranked:
            ch = self._chunks[idx]
            out.append(
                RetrievedChunk(
                    chunk_id=ch.metadata.chunk_id,
                    score=float(score),
                    content=ch.content,
                    metadata=ch.metadata,
                    trace={"retriever": "bgem3_sparse"},
                )
            )
        return out

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """执行 BM25 召回。

        BM25 这一层特别适合：
        - 错误码
        - 文号
        - 专有名词
        - 部门 / 班次 / 线别这类词面锚点强的问题
        """

        k = top_k or self._settings.bm25_top_k
        if not self._chunks:
            return []
        if self._bgem3.enabled:
            if getattr(self._settings, "vector_backend", "") == "milvus" and self._bgem3.get_function() is not None:
                hits = self._get_milvus().search_sparse(query, top_k=k, filters=filters)
                if hits:
                    return hits
            if self._bgem3.get_function() is not None and self._sparse_matrix is not None:
                return self._search_with_bgem3_sparse(query, top_k=k, filters=filters)
            logger.info(
                "BGEM3 unavailable for sparse retrieval; fallback to BM25",
                extra={"event": "sparse_backend_fallback", "fallback_backend": "bm25"},
            )
        if not self._bm25:
            return []
        q = tokenize(query)
        # `get_scores` 会给语料库中每个 chunk 一个相关性分数。
        scores = self._bm25.get_scores(q)
        ranked = sorted(
            (
                (idx, score)
                for idx, score in enumerate(scores)
                if chunk_matches_filters(self._chunks[idx].metadata, filters)
            ),
            key=lambda x: x[1],
            reverse=True,
        )[:k]
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
