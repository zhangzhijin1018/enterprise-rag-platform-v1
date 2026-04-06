"""门控策略单元测试模块。验证空召回或低置信度场景下是否正确触发拒答。"""

from core.config.settings import Settings
from core.orchestration.fusion_gate import fusion_results_actionable


def test_rrf_scores_not_blocked_by_min_retrieval() -> None:
    s = Settings.model_construct(fusion_strategy="rrf", min_retrieval_score=0.99)
    fused = [{"score": 0.02, "chunk_id": "a"}]
    assert fusion_results_actionable(s, fused) is True


def test_weighted_respects_min_retrieval() -> None:
    s = Settings.model_construct(fusion_strategy="weighted", min_retrieval_score=0.5)
    low = [{"score": 0.1, "chunk_id": "a"}]
    high = [{"score": 0.8, "chunk_id": "a"}]
    assert fusion_results_actionable(s, low) is False
    assert fusion_results_actionable(s, high) is True
