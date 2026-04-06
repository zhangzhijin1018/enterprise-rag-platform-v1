"""查询扩展预留模块。用于后续接入多查询扩展、同义词扩展或子问题分解。"""

from __future__ import annotations


def expand_queries(query: str, *, max_variants: int = 3) -> list[str]:
    """
    Multi-query expansion hook (placeholder).

    Returns additional query variants beyond the original string.
    """
    _ = max_variants
    return [query]
