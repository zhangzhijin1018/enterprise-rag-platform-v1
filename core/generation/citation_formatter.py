"""引用格式化模块。

负责在“检索命中对象”与“API / 前端可消费的引用结构”之间做转换。

这一层存在的核心意义是：
- 把内部 trace / metadata 收敛成用户能理解的引用字段
- 保证引用来源于真实命中的 chunk，而不是模型随意编造的文本片段
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.models.document import ChunkMetadata
from core.retrieval.schemas import RetrievedChunk


class Citation(BaseModel):
    """对外暴露的引用结构。

    这里的字段设计目标是“能解释答案来源”，所以会优先保留：
    - 定位信息：`doc_id / chunk_id / page / section / section_path`
    - 治理信息：`owner_department / data_classification / authority_level`
    - 排序信息：`retrieval_score / semantic_score / governance_rank_score`
    """

    doc_id: str = Field(description="文档 id")
    chunk_id: str = Field(description="chunk id")
    title: str = Field(description="文档标题")
    source: str = Field(description="来源文件或来源路径")
    page: int | None = Field(default=None, description="页码")
    section: str | None = Field(default=None, description="章节标题")
    doc_type: str | None = Field(default=None, description="文档类型")
    owner_department: str | None = Field(default=None, description="归属部门")
    data_classification: str | None = Field(default=None, description="数据分级")
    version: str | None = Field(default=None, description="版本号")
    effective_date: str | None = Field(default=None, description="生效日期")
    authority_level: str | None = Field(default=None, description="权威级别")
    source_system: str | None = Field(default=None, description="来源系统")
    business_domain: str | None = Field(default=None, description="业务域")
    process_stage: str | None = Field(default=None, description="流程阶段")
    section_path: str | None = Field(default=None, description="章节路径")
    matched_routes: list[str] = Field(default_factory=list, description="命中的检索路线")
    retrieval_score: float | None = Field(default=None, description="最终召回分")
    semantic_score: float | None = Field(default=None, description="语义相关分")
    governance_rank_score: float | None = Field(default=None, description="治理排序分")
    selection_reason: str | None = Field(default=None, description="为什么这条证据被选中")


def _selection_reason(hit: RetrievedChunk | None) -> str | None:
    """把 trace 里的排序信息收敛成对外可读的一句话。

    trace 本身更偏工程调试字段，
    这里把它压缩成一段人能直接读懂的说明，便于前端展示和 badcase 复盘。
    """

    if hit is None:
        return None

    trace = hit.trace
    reasons: list[str] = []
    matched_routes = [item for item in trace.get("matched_routes") or [] if isinstance(item, str)]
    if matched_routes:
        reasons.append("命中了 " + " / ".join(matched_routes) + " 检索路线")
    boost_reasons = [item for item in trace.get("metadata_boost_reasons") or [] if isinstance(item, str)]
    if boost_reasons:
        reasons.append("metadata 匹配增强：" + "、".join(boost_reasons[:3]))
    governance_bonus = trace.get("governance_bonus")
    if isinstance(governance_bonus, (int, float)) and governance_bonus > 0:
        reasons.append("治理排序提升")
    section_path = hit.metadata.extra_text("section_path")
    if section_path:
        reasons.append(f"定位章节：{section_path}")
    if not reasons:
        return None
    return "；".join(reasons)


def chunk_to_citation(item: RetrievedChunk | ChunkMetadata) -> Citation:
    """把 chunk / 检索结果转成引用对象。

    同时支持两类输入：
    - `RetrievedChunk`：有分数、有 trace，适合回答后展示
    - `ChunkMetadata`：只有来源信息，适合更基础的引用场景

    设计上优先从真实命中对象反推引用，而不是直接信任模型原始引用文本。
    """

    hit = item if isinstance(item, RetrievedChunk) else None
    meta = item.metadata if isinstance(item, RetrievedChunk) else item

    return Citation(
        doc_id=meta.doc_id,
        chunk_id=meta.chunk_id,
        title=meta.title,
        source=meta.source,
        page=meta.page,
        section=meta.section,
        doc_type=meta.extra_text("doc_type"),
        owner_department=meta.extra_text("owner_department"),
        data_classification=meta.extra_text("data_classification"),
        version=meta.extra_text("version"),
        effective_date=meta.extra_text("effective_date"),
        authority_level=meta.extra_text("authority_level"),
        source_system=meta.extra_text("source_system"),
        business_domain=meta.extra_text("business_domain"),
        process_stage=meta.extra_text("process_stage"),
        section_path=meta.extra_text("section_path"),
        matched_routes=[
            item for item in ((hit.trace.get("matched_routes") if hit else []) or []) if isinstance(item, str)
        ],
        retrieval_score=hit.score if hit is not None else None,
        semantic_score=(
            float(hit.trace["semantic_score"])
            if hit is not None and isinstance(hit.trace.get("semantic_score"), (int, float))
            else None
        ),
        governance_rank_score=(
            float(hit.trace["governance_rank_score"])
            if hit is not None and isinstance(hit.trace.get("governance_rank_score"), (int, float))
            else None
        ),
        selection_reason=_selection_reason(hit),
    )


def format_citations_from_chunks(chunks: list[RetrievedChunk]) -> list[Citation]:
    """把检索结果列表去重后转成引用列表。

    去重是必要的，因为同一个 chunk 可能：
    - 同时在 sparse / dense 命中
    - 在融合和重排后仍多次出现于中间过程
    """

    seen: set[str] = set()
    out: list[Citation] = []
    for c in chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        out.append(chunk_to_citation(c))
    return out


def citation_coverage(citations: list[Citation], retrieved_ids: list[str]) -> float:
    """计算“召回到的 chunk 里有多少真正被引用到了”。

    这个指标对评测很有价值：
    - 太低，说明模型可能没充分利用已检索证据
    - 太高也不一定更好，但至少可以帮助看“证据利用率”
    """

    if not retrieved_ids:
        return 0.0
    cited = {c.chunk_id for c in citations}
    return len(cited & set(retrieved_ids)) / max(1, len(set(retrieved_ids)))
