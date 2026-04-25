"""查询理解词典加载模块。

把高频业务词、场景词、设备词从硬编码规则里抽出来，
后续可以直接根据 badcase 调整词典，而不是总改正则源码。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.config.settings import Settings, get_settings


DEFAULT_QUERY_UNDERSTANDING_VOCAB: dict[str, Any] = {
    "department_suffixes": ["车间", "部门", "班组", "小组", "中心", "事业部", "工段", "科室", "仓库", "产线", "号线", "线"],
    "shift_keywords": ["白班", "夜班", "早班", "中班", "晚班"],
    "followup_prefixes": ["那", "那么", "如果是", "那如果", "这个", "那个", "它", "这条", "这项", "这类", "这种", "再", "然后"],
    "compare_keywords": ["和", "与", "vs", "VS", "对比", "比较", "区别", "差异"],
    "structured_fact_keywords": ["谁", "哪位", "哪里", "在哪", "几点", "多少", "名单", "电话", "联系人", "值班", "排班", "班次", "部门", "负责人", "归属"],
    "policy_keywords": ["制度", "规范", "标准", "办法", "规定", "条例"],
    "procedure_keywords": ["SOP", "流程", "步骤", "作业指导书", "操作手册", "巡检", "检修", "应急预案"],
    "meeting_keywords": ["会议纪要", "纪要", "决议", "讨论记录", "复盘"],
    "project_keywords": ["项目", "方案变更", "里程碑", "立项", "需求"],
    "equipment_keywords": ["锅炉", "汽轮机", "发电机", "输煤皮带", "磨煤机", "风机", "泵", "压缩机", "变压器", "阀门", "机组"],
    "system_keywords": ["接口", "平台", "系统", "数据库", "服务", "错误码", "API"],
    "sub_query_keywords": ["原因", "排查", "处理", "解决", "方案", "架构", "设计", "步骤", "流程", "以及", "并且", "影响", "区别", "对比"],
    "hyde_keywords": ["为什么", "原因", "原理", "方案", "架构", "设计", "最佳实践", "适用场景", "思路"],
    "business_domain_keywords": {
        "equipment_maintenance": ["巡检", "检修", "保养", "点检", "维修", "设备"],
        "dispatch": ["调度", "值班", "排班", "交接班"],
        "project_management": ["项目", "方案变更", "里程碑", "立项", "会议纪要"],
        "it_ops": ["接口", "错误码", "数据库", "平台", "系统", "发布"],
    },
    "department_aliases": {},
    "site_aliases": {},
    "system_aliases": {},
}
# 默认词典。
# 作用：
# 1. 即使没有外部 JSON 文件，query understanding 也能工作
# 2. 提供一套最小可用的中文企业问答词表
# 3. 后续外部词典会在它的基础上做 merge，而不是完全从零开始


@dataclass(frozen=True)
class QueryUnderstandingIndex:
    """查询理解词典的预编译内存索引。

    设计目标：
    - 避免每次请求重复读取 JSON
    - 避免每次请求重复编译 regex
    - 把 alias / business domain 的原始 dict 先整理成更适合匹配的结构
    """

    vocab: dict[str, Any]
    scene_patterns: dict[str, re.Pattern[str]]
    department_pattern: re.Pattern[str]
    equipment_pattern: re.Pattern[str]
    business_domain_entries: tuple[tuple[str, tuple[str, ...]], ...]
    department_alias_entries: tuple[tuple[str, tuple[str, ...]], ...]
    site_alias_entries: tuple[tuple[str, tuple[str, ...]], ...]
    system_alias_entries: tuple[tuple[str, tuple[str, ...]], ...]


def _keyword_regex(keywords: list[str], *, anchored_start: bool = False) -> re.Pattern[str]:
    """把关键词列表编译成 regex。"""

    escaped = [re.escape(item) for item in keywords if item]
    if not escaped:
        pattern = r"$^"
    else:
        pattern = "|".join(sorted(escaped, key=len, reverse=True))
    if anchored_start:
        return re.compile(rf"^(?:{pattern})")
    return re.compile(rf"(?:{pattern})", re.IGNORECASE)


def _department_regex(suffixes: list[str]) -> re.Pattern[str]:
    """根据部门后缀词构建部门识别 regex。"""

    token_chars = r"[\u4e00-\u9fffA-Za-z0-9_-]"
    escaped = [re.escape(item) for item in suffixes if item]
    suffix_pattern = "|".join(sorted(escaped, key=len, reverse=True)) or r"$^"
    return re.compile(rf"(({token_chars}{{2,30}}(?:{suffix_pattern})))")


def _equipment_regex(keywords: list[str]) -> re.Pattern[str]:
    """根据设备词构建设备识别 regex。"""

    token_chars = r"[\u4e00-\u9fffA-Za-z0-9_-]"
    escaped = [re.escape(item) for item in keywords if item]
    pattern = "|".join(sorted(escaped, key=len, reverse=True)) or r"$^"
    return re.compile(rf"((?:{token_chars}{{0,30}})?(?:{pattern}))")


def _scene_regex_map(vocab: dict[str, Any]) -> dict[str, re.Pattern[str]]:
    """把词典里的场景词编译成一组 scene pattern。"""

    return {
        "policy": _keyword_regex(list(vocab.get("policy_keywords") or [])),
        "procedure": _keyword_regex(list(vocab.get("procedure_keywords") or [])),
        "meeting": _keyword_regex(list(vocab.get("meeting_keywords") or [])),
        "project": _keyword_regex(list(vocab.get("project_keywords") or [])),
        "system": _keyword_regex(list(vocab.get("system_keywords") or [])),
        "compare": _keyword_regex(list(vocab.get("compare_keywords") or [])),
        "structured_fact": _keyword_regex(list(vocab.get("structured_fact_keywords") or [])),
        "sub_query": _keyword_regex(list(vocab.get("sub_query_keywords") or [])),
        "hyde": _keyword_regex(list(vocab.get("hyde_keywords") or [])),
        "followup": _keyword_regex(list(vocab.get("followup_prefixes") or []), anchored_start=True),
        "shift": _keyword_regex(list(vocab.get("shift_keywords") or [])),
    }


def _sorted_mapping_entries(mapping: dict[str, Any]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """把别名字典或业务域词典整理成稳定顺序的元组结构。

    这里按“最长 alias 优先”排序，目的是：
    - 降低短词提前命中造成的误判
    - 提高规范实体归一时的稳定性
    """

    entries: list[tuple[str, tuple[str, ...]]] = []
    for key, values in mapping.items():
        normalized_values = tuple(str(item).strip() for item in values or [] if str(item).strip())
        if not str(key).strip() or not normalized_values:
            continue
        entries.append((str(key).strip(), normalized_values))
    entries.sort(key=lambda item: (-max(len(alias) for alias in item[1]), item[0]))
    return tuple(entries)


def _merge_vocab(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """把外部词典覆盖到默认词典上。"""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _normalize_vocab(raw: dict[str, Any]) -> dict[str, Any]:
    """把词典值规整成干净、稳定的字符串结构。"""

    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            normalized[key] = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, dict):
            normalized[key] = {
                str(sub_key).strip(): [str(item).strip() for item in sub_value if str(item).strip()]
                for sub_key, sub_value in value.items()
                if str(sub_key).strip() and isinstance(sub_value, list)
            }
        else:
            normalized[key] = value
    return normalized


@lru_cache(maxsize=8)
def _load_vocab_cached(path_text: str) -> dict[str, Any]:
    """读取并缓存原始词典。"""

    base = _normalize_vocab(DEFAULT_QUERY_UNDERSTANDING_VOCAB)
    if not path_text:
        return base
    path = Path(path_text)
    if not path.is_file():
        return base
    try:
        override = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return base
    if not isinstance(override, dict):
        return base
    return _merge_vocab(base, _normalize_vocab(override))


def load_query_understanding_vocab(settings: Settings | None = None) -> dict[str, Any]:
    """加载查询理解词典原始结构。"""

    cfg = settings or get_settings()
    return _load_vocab_cached(str(cfg.query_understanding_vocab_path))


@lru_cache(maxsize=8)
def _build_index_cached(path_text: str) -> QueryUnderstandingIndex:
    """构建并缓存查询理解词典索引。"""

    vocab = _load_vocab_cached(path_text)
    return QueryUnderstandingIndex(
        vocab=vocab,
        scene_patterns=_scene_regex_map(vocab),
        department_pattern=_department_regex(list(vocab.get("department_suffixes") or [])),
        equipment_pattern=_equipment_regex(list(vocab.get("equipment_keywords") or [])),
        business_domain_entries=_sorted_mapping_entries(
            dict(vocab.get("business_domain_keywords") or {})
        ),
        department_alias_entries=_sorted_mapping_entries(dict(vocab.get("department_aliases") or {})),
        site_alias_entries=_sorted_mapping_entries(dict(vocab.get("site_aliases") or {})),
        system_alias_entries=_sorted_mapping_entries(dict(vocab.get("system_aliases") or {})),
    )


def load_query_understanding_index(settings: Settings | None = None) -> QueryUnderstandingIndex:
    """加载查询理解词典索引。

    这是 query understanding 运行时更推荐使用的入口：
    - 词典文件仍然来自 JSON
    - 但实际请求侧尽量直接使用这个预编译索引
    """

    cfg = settings or get_settings()
    return _build_index_cached(str(cfg.query_understanding_vocab_path))
