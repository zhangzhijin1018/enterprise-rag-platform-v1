"""交叉编码器重排模块。负责对候选片段做更精细的相关性打分。"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from core.config.settings import Settings, get_settings
from core.retrieval.schemas import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._settings.reranker_model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        n = top_n or self._settings.rerank_top_n
        if not candidates:
            return []
        pairs = [[query, c.content] for c in candidates]
        scores = self._get_model().predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:n]
        out: list[RetrievedChunk] = []
        for ch, sc in ranked:
            out.append(
                RetrievedChunk(
                    chunk_id=ch.chunk_id,
                    score=float(sc),
                    content=ch.content,
                    metadata=ch.metadata,
                    trace={**ch.trace, "reranker": "cross_encoder"},
                )
            )
        return out
