"""融合结果门控模块。根据召回结果数量与分数决定是否继续进入重排和生成阶段。"""

from __future__ import annotations

from typing import Any

from core.config.settings import Settings


def fusion_results_actionable(settings: Settings, fused_hits: list[dict[str, Any]]) -> bool:
    """
    RRF scores are on a much smaller scale than weighted fusion; only apply
    MIN_RETRIEVAL_SCORE when using weighted fusion.
    """
    if not fused_hits:
        return False
    if settings.fusion_strategy != "weighted":
        return True
    try:
        best = max(float(x.get("score", 0.0)) for x in fused_hits)
    except ValueError:
        best = 0.0
    return best >= settings.min_retrieval_score
