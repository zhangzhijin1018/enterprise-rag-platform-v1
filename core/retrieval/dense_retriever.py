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

    def _get_model(self) -> SentenceTransformer:
        """懒加载 embedding 模型。"""

        if self._model is None:
            self._model = SentenceTransformer(self._settings.embedding_model_name)
        return self._model

    def rebuild(self, chunks: Sequence[TextChunk], matrix: np.ndarray | None) -> None:
        """重建检索所需的文档向量矩阵。"""

        self._chunks = list(chunks)
        if not self._chunks:
            self._matrix = None
            return
        self._matrix = matrix
        if self._matrix is None and self._chunks:
            # 如果上游没有传现成向量，就现场对全部 chunk 重新编码。
            texts = [c.content for c in self._chunks]
            self._matrix = self._get_model().encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )

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
