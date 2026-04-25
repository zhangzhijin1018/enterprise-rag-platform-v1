"""检索 metadata filter 辅助函数。

这个模块负责解决一个很实际的问题：

- 上游传进来的过滤条件是“业务语义”
- 底层向量库 / 检索器认识的是“字段表达式”

所以这里做的是一层“过滤条件翻译”：
- 能下推到 Milvus 的，尽量下推，减少无效召回
- 不能直接下推的，保留给上层 post-filter
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from core.models.document import ChunkMetadata

# 这些字段是当前允许直接下推到 Milvus schema 的一级过滤字段。
# 也就是说：
# - 命中这里的字段，可以直接变成 Milvus filter expression
# - 不在这里的字段，通常要走上层 post-filter
MILVUS_DIRECT_FILTER_FIELDS = {
    "doc_id",
    "source",
    "title",
    "section",
    "page",
    "chunk_level",
    "doc_number",
    "department",
    "owner_department",
    "group_company",
    "subsidiary",
    "plant",
    "shift",
    "line",
    "person",
    "time",
    "environment",
    "version",
    "version_status",
    "doc_category",
    "doc_type",
    "status",
    "data_classification",
    "effective_date",
    "expiry_date",
    "authority_level",
    "source_system",
    "issued_by",
    "approved_by",
    "owner_role",
    "business_domain",
    "process_stage",
    "applicable_region",
    "applicable_site",
    "equipment_type",
    "equipment_id",
    "system_name",
    "project_name",
    "project_phase",
    "section_path",
    "section_level",
    "section_type",
    "contains_table",
    "contains_steps",
    "contains_contact",
    "contains_version_signal",
    "contains_risk_signal",
}

# 同义实体组。
# 组内字段表示同一个业务语义在不同 metadata 字段上的落点，
# 检索时使用“组内 OR，组间 AND”的语义，避免过滤条件过严。
FILTER_OR_GROUPS: dict[str, tuple[str, ...]] = {
    "department_scope": ("department", "owner_department"),
    "site_scope": ("plant", "applicable_site"),
}


def _normalize_scalar(value: Any) -> str:
    """把任意值规整成可比较的小写字符串。"""

    return str(value).strip().casefold()


def _metadata_value(metadata: ChunkMetadata, key: str) -> Any:
    """优先读取一级字段，否则退回 extra。

    这样做是为了兼容两类字段：
    - `page / section / title` 这类一级字段
    - `department / business_domain / allowed_roles` 这类扩展字段
    """

    if hasattr(metadata, key):
        return getattr(metadata, key)
    return metadata.extra.get(key)


def _metadata_text(metadata: ChunkMetadata) -> str:
    """把 metadata 压平成一个可做保守子串匹配的文本。

    这是兜底策略：
    - 最优先还是结构化字段精确匹配
    - 只有字段对不上时，才退化成保守子串匹配

    好处是 recall 更稳，坏处是精度没结构化匹配高，所以这里只做 fallback。
    """

    parts = [
        metadata.doc_id,
        metadata.chunk_id,
        metadata.source,
        metadata.title,
        metadata.section or "",
        json.dumps(metadata.extra, ensure_ascii=False, sort_keys=True),
    ]
    return " ".join(part for part in parts if part).casefold()


def _actual_values(value: Any) -> list[str]:
    """把 metadata 实际值规整成可比较列表。"""

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [_normalize_scalar(item) for item in value if _normalize_scalar(item)]
    return [_normalize_scalar(value)]


def _grouped_filter_items(filters: Mapping[str, Any]) -> list[tuple[tuple[str, ...], list[Any]]]:
    """把同义实体过滤条件收敛成“组内 OR，组间 AND”的形式。

    例如部门相关语义可能落在：
    - `department`
    - `owner_department`

    如果直接逐字段 AND，很多本来应该命中的 chunk 会被误杀。
    所以这里把它们视作同一业务语义组，采用：
    - 组内 OR
    - 组间 AND
    """

    grouped: list[tuple[tuple[str, ...], list[Any]]] = []
    consumed_keys: set[str] = set()

    for field_group in FILTER_OR_GROUPS.values():
        merged_values: list[Any] = []
        seen_values: set[str] = set()
        for key in field_group:
            if key not in filters:
                continue
            consumed_keys.add(key)
            raw_values = filters[key] if isinstance(filters[key], list) else [filters[key]]
            for value in raw_values:
                value_key = repr(value)
                if value_key in seen_values:
                    continue
                seen_values.add(value_key)
                merged_values.append(value)
        if merged_values:
            grouped.append((field_group, merged_values))

    for key, value in filters.items():
        if key in consumed_keys:
            continue
        values = value if isinstance(value, list) else [value]
        grouped.append(((key,), list(values)))

    return grouped


def chunk_matches_filters(metadata: ChunkMetadata, filters: Mapping[str, Any] | None) -> bool:
    """判断 chunk metadata 是否满足结构化过滤条件。

    当前策略尽量保守：
    - 优先匹配一级 metadata 字段与 extra 扩展字段
    - 若没有精确字段，再退化为 metadata 文本子串匹配

    这个函数通常用于：
    - 本地 post-filter
    - Milvus 无法直接表达的过滤兜底
    - 单测里验证过滤语义是否符合预期
    """

    if not filters:
        return True

    metadata_text = _metadata_text(metadata)
    for field_group, expected_values in _grouped_filter_items(filters):
        matched = False
        actual_values: list[str] = []
        for key in field_group:
            actual_values.extend(_actual_values(_metadata_value(metadata, key)))
        for expected in expected_values:
            expected_norm = _normalize_scalar(expected)
            if not expected_norm:
                continue
            if expected_norm in actual_values:
                matched = True
                break
            if expected_norm in metadata_text:
                matched = True
                break
        if not matched:
            return False
    return True


def build_milvus_filter_expression(filters: Mapping[str, Any] | None) -> str:
    """把可直接映射到 Milvus schema 的过滤条件转成表达式。

    当前优先下推这些一级字段：
    - 基础检索字段：`doc_id / source / title / section / page / chunk_level`
    - 企业事实字段：`department / shift / line / person / time / environment / version / doc_category`

    不在 schema 中的字段仍走上层 post-filter。

    默认总会加上 `searchable == true`，原因是：
    - parent chunk 虽然会入库，但不应该直接参与召回
    - 这个过滤条件可以保证底层优先只搜 child chunk
    """

    clauses = ["searchable == true"]
    if not filters:
        return " and ".join(clauses)

    for field_group, values in _grouped_filter_items(filters):
        direct_fields = [key for key in field_group if key in MILVUS_DIRECT_FILTER_FIELDS]
        if not direct_fields:
            continue
        normalized_values = [str(item).strip() for item in values if str(item).strip()]
        if not normalized_values:
            continue
        if len(direct_fields) == 1 and direct_fields[0] == "page":
            page_values = [str(int(float(item))) for item in normalized_values]
            clauses.append(f'page in [{", ".join(page_values)}]')
            continue
        escaped = [item.replace("\\", "\\\\").replace('"', '\\"') for item in normalized_values]
        field_clauses: list[str] = []
        for key in direct_fields:
            if len(escaped) == 1:
                field_clauses.append(f'{key} == "{escaped[0]}"')
            else:
                field_clauses.append(f'{key} in ["' + '","'.join(escaped) + '"]')
        if len(field_clauses) == 1:
            clauses.append(field_clauses[0])
        else:
            clauses.append("(" + " or ".join(field_clauses) + ")")
    return " and ".join(clauses)
