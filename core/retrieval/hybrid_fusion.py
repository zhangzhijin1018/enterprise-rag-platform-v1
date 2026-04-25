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
    """执行归一化后的加权融合。

    适用场景：
    - 我们大致知道 sparse / dense 哪一路更可信
    - 或者不同 query_scene 下，希望显式偏向某一路

    所以它比 RRF 更“可控”，但前提是你得对权重有一定把握。
    """

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
    """根据配置选择具体融合策略。

    这里把融合本身独立出来的好处是：
    - retrieval 节点不用关心具体算法细节
    - 后续换成别的 fusion 方法时改动面很小
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._weighted_query_scenes = {
            item.strip()
            for item in str(self._settings.weighted_fusion_query_scenes or "").split(",")
            if item.strip()
        }
        self._scene_weights = self._parse_scene_weights(self._settings.weighted_fusion_scene_weights)

    def _parse_scene_weights(self, raw: str | None) -> dict[str, float]:
        """解析 `scene:weight` 形式的场景权重配置。

        这样做的目的是把“不同 query_scene 用不同 sparse 权重”配置化，
        避免把业务策略硬编码在 retrieval 节点里。
        """

        out: dict[str, float] = {}
        for item in str(raw or "").split(","):
            pair = item.strip()
            if not pair or ":" not in pair:
                continue
            scene, weight_text = pair.split(":", 1)
            scene = scene.strip()
            if not scene:
                continue
            try:
                weight = float(weight_text.strip())
            except ValueError:
                continue
            out[scene] = min(max(weight, 0.0), 1.0)
        return out

    def _resolve_strategy(self, query_scene: str | None = None) -> str:
        """根据 query_scene 决定当前请求实际使用的融合策略。

        例如：
        - 制度号、错误码这类词面强约束查询，更适合 weighted
        - 普通开放问句，默认继续使用全局融合策略
        """

        scene = str(query_scene or "").strip()
        if scene and scene in self._weighted_query_scenes:
            return "weighted"
        return self._settings.fusion_strategy

    def _resolve_sparse_weight(self, query_scene: str | None = None) -> float:
        """根据 query_scene 决定 weighted fusion 的 sparse 权重。

        sparse 权重越高，词面命中的影响越大；
        越低，则表示更相信 dense 的语义相似度。
        """

        scene = str(query_scene or "").strip()
        if scene and scene in self._scene_weights:
            return self._scene_weights[scene]
        return self._settings.fusion_sparse_weight

    def resolve_policy(self, query_scene: str | None = None) -> tuple[str, float]:
        """返回当前 query_scene 对应的融合策略与 sparse 权重。

        这个方法的作用是把“策略决策”从“具体融合执行”里拆出来，
        方便：
        - 项目上层自己的 `fuse(...)`
        - Milvus 原生 `hybrid_search + ranker`
        共享同一套策略来源，避免两边出现口径漂移。
        """

        strategy = self._resolve_strategy(query_scene=query_scene)
        sparse_weight = self._resolve_sparse_weight(query_scene=query_scene)
        return strategy, sparse_weight

    def fuse(
        self,
        sparse_hits: list[RetrievedChunk],
        dense_hits: list[RetrievedChunk],
        *,
        query_scene: str | None = None,
    ) -> list[RetrievedChunk]:
        """融合 BM25 与向量召回结果。

        这是 retrieval 主链路里“把两路候选合并成一份候选集”的关键步骤。
        返回结果里会额外补 `trace`，方便后续看到：
        - 用了哪种 fusion
        - sparse 权重是多少
        - 当前 query_scene 是什么
        """

        top_k = self._settings.hybrid_top_k
        strategy, sparse_weight = self.resolve_policy(query_scene=query_scene)
        if strategy == "weighted":
            fused = weighted_fusion(
                sparse_hits,
                dense_hits,
                sparse_weight,
                top_k,
            )
        else:
            fused = reciprocal_rank_fusion([sparse_hits, dense_hits], top_k=top_k)

        out: list[RetrievedChunk] = []
        for item in fused:
            out.append(
                RetrievedChunk(
                    chunk_id=item.chunk_id,
                    score=item.score,
                    content=item.content,
                    metadata=item.metadata,
                    trace={
                        **item.trace,
                        "fusion_strategy": strategy,
                        "fusion_sparse_weight": sparse_weight,
                        "query_scene": query_scene or "",
                    },
                )
            )
        return out
