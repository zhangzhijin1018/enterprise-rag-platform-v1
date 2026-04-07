"""向量检索模块。

原理简述：
- 先用 embedding 模型把文档和查询编码到同一个向量空间；
- 再通过向量点积 / 余弦相似度找到最接近的问题相关 chunk；
- 适合召回“词不完全匹配但语义相近”的内容。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config.settings import Settings, get_settings
from core.models.document import TextChunk
from core.retrieval.schemas import RetrievedChunk


class DenseRetriever:
    """基于句向量的稠密检索器。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: SentenceTransformer | None = None
        self._chunks: list[TextChunk] = []
        self._matrix: np.ndarray | None = None

    @property
    def backend_name(self) -> str:
        """返回当前 dense backend 名称。"""

        return "file"

    def _get_model(self) -> SentenceTransformer:
        """懒加载 embedding 模型。"""

        if self._model is None:
            self._model = SentenceTransformer(self._settings.embedding_model_name)
        return self._model

    def rebuild(self, chunks: Sequence[TextChunk], matrix: np.ndarray | None) -> None:
        """重建检索所需的文档向量矩阵。

        和稀疏检索器一样，第三轮增强后这里只会把 `searchable=True` 的 chunk，
        也就是默认的 child chunks，放进向量召回矩阵。

        注意这里和 `IndexStore` 的职责分工：

        - `IndexStore` 仍然持有全部 parent + child chunks
        - `DenseRetriever` 只把 child chunks 放进搜索矩阵

        这样既能保留 parent 回扩能力，又能避免长 parent chunk 直接参与 dense retrieval，
        导致召回目标变得过粗。
        """

        self._chunks = [c for c in chunks if c.searchable]
        if not self._chunks:
            self._matrix = None
            return
        self._matrix = None
        if matrix is not None:
            # `matrix` 和 `chunks` 在 IndexStore 中是一一对齐的全量矩阵。
            # 这里需要把 parent rows 过滤掉，只留下 searchable child rows。
            child_indices = [idx for idx, chunk in enumerate(chunks) if chunk.searchable]
            self._matrix = matrix[child_indices] if child_indices else None
        if self._matrix is None and self._chunks:
            # 如果上游没有传现成向量，就现场对全部 chunk 重新编码。
            texts = [c.content for c in self._chunks]
            self._matrix = self._get_model().encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )

    def ensure_remote_index(self, chunks: Sequence[TextChunk], matrix: np.ndarray | None) -> None:
        """给外部向量数据库预留的补建钩子。

        文件型 backend 不需要远端补建，所以默认是 no-op。
        """

    def sync_remote_index(self, chunks: Sequence[TextChunk], matrix: np.ndarray | None) -> None:
        """给外部向量数据库预留的显式同步钩子。

        文件型 backend 没有远端集合，所以默认是 no-op。
        """

    def embed_query(self, query: str) -> np.ndarray:
        """把查询编码成单位向量。

        这里开启 `normalize_embeddings=True`，意味着后续点积就等价于余弦相似度。
        """

        return self._get_model().encode([query], normalize_embeddings=True, show_progress_bar=False)[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """把多段文档文本编码成向量矩阵。"""

        if not texts:
            # 空数组时返回固定维度的零矩阵，方便上游保持统一数据类型。
            return np.zeros((0, 384), dtype=np.float32)
        return self._get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """执行向量召回。"""

        k = top_k or self._settings.dense_top_k
        if not self._chunks or self._matrix is None:
            return []
        q = self.embed_query(query)
        # 因为向量已归一化，所以这里的矩阵乘法就是批量计算余弦相似度。
        sims = self._matrix @ q
        ranked = np.argsort(-sims)[:k]
        out: list[RetrievedChunk] = []
        for idx in ranked:
            ch = self._chunks[int(idx)]
            out.append(
                RetrievedChunk(
                    chunk_id=ch.metadata.chunk_id,
                    score=float(sims[int(idx)]),
                    content=ch.content,
                    metadata=ch.metadata,
                    trace={"retriever": "dense"},
                )
            )
        return out
