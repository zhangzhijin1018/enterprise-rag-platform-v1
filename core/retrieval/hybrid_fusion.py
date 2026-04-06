"""混合召回融合模块。

本项目支持两种常见融合策略：
- RRF（Reciprocal Rank Fusion）：按排名做融合，不依赖不同检索器的分数尺度一致。
- weighted fusion：先归一化分数，再按权重线性组合，更适合明确知道两路检索相对重要性时使用。
"""

from __future__ import annotations

from collections import defaultdict

from core.config.settings import Settings, get_settings
from core.retrieval.schemas import RetrievedChunk


def reciprocal_rank_fusion(
    lists: list[list[RetrievedChunk]],
    k: int = 60,
    top_k: int = 30,
) -> list[RetrievedChunk]:
    """执行 RRF 融合。

    公式直觉：
    - 一个 chunk 在多路召回里排名越靠前，累计分越高。
    - `k` 越大，前几名之间的分差越平滑。

    示例：
    如果某 chunk 在 BM25 排第 1、在 dense 排第 3，
    它的分数约为 `1/(60+1) + 1/(60+3)`。
    """

    scores: dict[str, float] = defaultdict(float)
    best: dict[str, RetrievedChunk] = {}
    for lst in lists:
        for rank, item in enumerate(lst, start=1):
            cid = item.chunk_id
            scores[cid] += 1.0 / (k + rank)
            if cid not in best or item.score > best[cid].score:
                best[cid] = item
    ordered = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:top_k]
    out: list[RetrievedChunk] = []
    for cid in ordered:
        base = best[cid]
        out.append(
            RetrievedChunk(
                chunk_id=base.chunk_id,
                score=float(scores[cid]),
                content=base.content,
                metadata=base.metadata,
                trace={**base.trace, "fusion": "rrf"},
            )
        )
    return out


def weighted_fusion(
    sparse: list[RetrievedChunk],
    dense: list[RetrievedChunk],
    sparse_weight: float,
    top_k: int,
) -> list[RetrievedChunk]:
    """执行归一化后的加权融合。"""

    def norm(items: list[RetrievedChunk]) -> dict[str, float]:
        if not items:
            return {}
        # 不同检索器的原始分数分布不同，先除以当前列表里的最大绝对值做粗归一化。
        mx = max(abs(i.score) for i in items) or 1.0
        return {i.chunk_id: i.score / mx for i in items}

    s_map = norm(sparse)
    d_map = norm(dense)
    w_d = 1.0 - sparse_weight
    keys = set(s_map) | set(d_map)
    scores = {cid: sparse_weight * s_map.get(cid, 0.0) + w_d * d_map.get(cid, 0.0) for cid in keys}
    best: dict[str, RetrievedChunk] = {}
    for it in sparse + dense:
        cid = it.chunk_id
        if cid not in best:
            best[cid] = it
    ordered = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:top_k]
    out: list[RetrievedChunk] = []
    for cid in ordered:
        base = best[cid]
        out.append(
            RetrievedChunk(
                chunk_id=base.chunk_id,
                score=float(scores[cid]),
                content=base.content,
                metadata=base.metadata,
                trace={**base.trace, "fusion": "weighted"},
            )
        )
    return out


class HybridFusion:
    """根据配置选择具体融合策略。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def fuse(
        self,
        sparse_hits: list[RetrievedChunk],
        dense_hits: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """融合 BM25 与向量召回结果。"""

        top_k = self._settings.hybrid_top_k
        if self._settings.fusion_strategy == "weighted":
            return weighted_fusion(
                sparse_hits,
                dense_hits,
                self._settings.fusion_sparse_weight,
                top_k,
            )
        return reciprocal_rank_fusion([sparse_hits, dense_hits], top_k=top_k)
