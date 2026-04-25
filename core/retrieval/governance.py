"""企业知识治理排序与冲突检测辅助模块。"""

from __future__ import annotations

from datetime import date
import re
from typing import Iterable

from core.models.document import ChunkMetadata
from core.retrieval.schemas import RetrievedChunk

_AUTHORITY_RANK = {
    "low": 1,
    "medium": 2,
    "high": 3,
}
# 权威级别映射表。
# 数值越大，说明该证据在企业治理视角下越值得优先采用。


def _parse_version(value: str | None) -> tuple[int, ...]:
    """把版本字符串解析成可比较的整数元组。"""

    if not value:
        return ()
    parts = re.findall(r"\d+", value)
    return tuple(int(part) for part in parts)


def _parse_date(value: str | None) -> date | None:
    """把日期字符串解析成 `date`。"""

    if not value:
        return None
    text = value.strip().replace("/", "-")
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _authority_rank(metadata: ChunkMetadata) -> int:
    """读取 metadata 中的权威级别并映射成排序数值。"""

    return _AUTHORITY_RANK.get((metadata.extra_text("authority_level") or "").lower(), 0)


def _version_rank(metadata: ChunkMetadata) -> tuple[int, ...]:
    """读取 metadata 中的版本号并转成可比较形式。"""

    return _parse_version(metadata.extra_text("version"))


def _effective_date_rank(metadata: ChunkMetadata) -> int:
    """读取 metadata 中的生效日期并转成可比较序号。"""

    parsed = _parse_date(metadata.extra_text("effective_date"))
    return parsed.toordinal() if parsed else 0


def _normalize_metric(values: list[int]) -> list[float]:
    """把整数指标归一化到 0-1 区间。"""

    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return [0.0 for _ in values]
    span = maximum - minimum
    return [(value - minimum) / span for value in values]


def _normalize_version_metric(values: list[tuple[int, ...]]) -> list[float]:
    """把版本号这种元组指标归一化到 0-1 区间。"""

    if not values:
        return []
    if len({value for value in values}) <= 1:
        return [0.0 for _ in values]
    ordered = {value: idx for idx, value in enumerate(sorted(set(values)))}
    maximum = max(ordered.values()) or 1
    return [ordered[value] / maximum for value in values]


def apply_governance_ranking(
    hits: list[RetrievedChunk],
    settings: object,
) -> list[RetrievedChunk]:
    """在语义重排结果之上叠加企业治理优先级。

    当前主要看 3 个治理信号：
    - `authority_level`：权威级别
    - `effective_date`：生效日期
    - `version`：版本号

    语义重排负责“相关不相关”，治理排序负责“同样相关时优先信谁”。
    """

    if len(hits) <= 1 or not getattr(settings, "enable_governance_ranking", True):
        return hits

    authority_values = [_authority_rank(hit.metadata) for hit in hits]
    freshness_values = [_effective_date_rank(hit.metadata) for hit in hits]
    version_values = [_version_rank(hit.metadata) for hit in hits]

    authority_norm = _normalize_metric(authority_values)
    freshness_norm = _normalize_metric(freshness_values)
    version_norm = _normalize_version_metric(version_values)

    ranked: list[tuple[float, int, RetrievedChunk]] = []
    for index, hit in enumerate(hits):
        authority_bonus = authority_norm[index] * float(
            getattr(settings, "authority_priority_boost", 0.0)
        )
        freshness_bonus = freshness_norm[index] * float(
            getattr(settings, "freshness_priority_boost", 0.0)
        )
        version_bonus = version_norm[index] * float(
            getattr(settings, "version_priority_boost", 0.0)
        )
        governance_bonus = authority_bonus + freshness_bonus + version_bonus
        trace = dict(hit.trace)
        # `semantic_score` 保留 cross-encoder 或上游语义阶段的原始排序分，
        # `governance_rank_score` 表示叠加企业治理 bonus 之后的最终排序分。
        trace["semantic_score"] = hit.score
        trace["governance_bonus"] = round(governance_bonus, 6)
        trace["governance_rank_score"] = round(hit.score + governance_bonus, 6)
        trace["authority_level"] = hit.metadata.extra_text("authority_level")
        trace["effective_date"] = hit.metadata.extra_text("effective_date")
        trace["version"] = hit.metadata.extra_text("version")
        ranked.append(
            (
                hit.score + governance_bonus,
                -index,
                hit.model_copy(update={"trace": trace}),
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked]


def _topic_key(metadata: ChunkMetadata) -> str:
    """收敛出“同主题文档”的分组键。"""

    title = (metadata.title or "").strip().casefold()
    if title:
        return title
    source = (metadata.source or "").strip().casefold()
    if source:
        return source
    return metadata.doc_id.strip().casefold()


def _distinct_non_empty(values: Iterable[str | None]) -> list[str]:
    """去重并保留非空字符串。"""

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = (value or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _describe_hit(hit: RetrievedChunk) -> str:
    """把命中的证据压成适合冲突摘要展示的一行文本。"""

    metadata = hit.metadata
    title = metadata.title or metadata.source or metadata.doc_id
    details: list[str] = []
    version = metadata.extra_text("version")
    effective_date = metadata.extra_text("effective_date")
    authority = metadata.extra_text("authority_level")
    if version:
        details.append(f"版本 {version}")
    if effective_date:
        details.append(f"生效日期 {effective_date}")
    if authority:
        details.append(f"权威级别 {authority}")
    if details:
        return f"{title}（{'，'.join(details)}）"
    return title


def detect_document_conflicts(
    hits: list[RetrievedChunk],
    settings: object,
) -> tuple[bool, str]:
    """检测多版本 / 多权威候选是否需要显式提示。

    这里不是做通用语义矛盾检测，而是做结构化冲突检测：
    - 版本不同
    - 生效日期不同
    - 权威级别不同
    """

    if not hits or not getattr(settings, "enable_conflict_detection", True):
        return False, ""

    top_k = max(2, int(getattr(settings, "conflict_detection_top_k", 5)))
    candidates = hits[:top_k]
    groups: dict[str, list[RetrievedChunk]] = {}
    for hit in candidates:
        groups.setdefault(_topic_key(hit.metadata), []).append(hit)

    for group in groups.values():
        if len(group) < 2:
            continue
        versions = _distinct_non_empty(hit.metadata.extra_text("version") for hit in group)
        effective_dates = _distinct_non_empty(
            hit.metadata.extra_text("effective_date") for hit in group
        )
        authority_levels = _distinct_non_empty(
            hit.metadata.extra_text("authority_level") for hit in group
        )

        reasons: list[str] = []
        if len(versions) > 1:
            reasons.append(f"版本不一致（{', '.join(versions)}）")
        if len(effective_dates) > 1:
            reasons.append(f"生效日期不一致（{', '.join(effective_dates)}）")
        if len(authority_levels) > 1:
            reasons.append(f"权威级别不同（{', '.join(authority_levels)}）")
        if not reasons:
            continue
        preferred = group[0]
        return (
            True,
            "命中多份同主题证据，"
            + "；".join(reasons)
            + f"；当前已优先采用 {_describe_hit(preferred)}。",
        )
    return False, ""
