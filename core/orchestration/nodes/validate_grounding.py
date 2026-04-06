"""落地性校验节点模块。负责给最终答案打 grounded / refusal 等收尾标记。"""

from __future__ import annotations

from core.config.settings import get_settings
from core.generation.citation_formatter import Citation
from core.observability.metrics import CITATION_COVERAGE
from core.orchestration.state import RAGState
from core.retrieval.schemas import RetrievedChunk


async def validate_grounding_node(state: RAGState) -> RAGState:
    settings = get_settings()
    citations_raw = state.get("citations") or []
    citations = [Citation.model_validate(c) for c in citations_raw]
    ctx_raw = state.get("reranked_hits") or []
    contexts = [RetrievedChunk.model_validate(x) for x in ctx_raw]
    retrieved_ids = [c.chunk_id for c in contexts]

    grounding_ok = True
    conf = float(state.get("confidence") or 0.0)
    refusal = bool(state.get("refusal"))

    if not refusal and not citations:
        grounding_ok = False
        conf = min(conf, 0.25)
        refusal = True

    cited_ids = {c.chunk_id for c in citations}
    allowed = set(retrieved_ids)
    if cited_ids - allowed:
        grounding_ok = False
        conf = min(conf, 0.2)

    if conf < settings.refusal_confidence_threshold and not refusal:
        refusal = True
        grounding_ok = False

    cov = len(cited_ids & set(retrieved_ids)) / max(1, len(set(retrieved_ids)))
    CITATION_COVERAGE.observe(cov)

    prev_reason = (state.get("refusal_reason") or "").strip()
    if refusal and not prev_reason:
        prev_reason = "low_confidence"

    return {
        "grounding_ok": grounding_ok,
        "confidence": conf,
        "refusal": refusal,
        "refusal_reason": prev_reason,
        "citations": [c.model_dump(mode="json") for c in citations],
    }
