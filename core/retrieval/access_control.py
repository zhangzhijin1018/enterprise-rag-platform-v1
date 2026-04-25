"""企业访问控制与数据分级基础模块。

第一阶段只实现基于 chunk metadata 的轻量规则：
- 从用户上下文构造访问过滤条件
- 判断单个 chunk 是否可访问
- 结合数据分级推导模型路由标签
- 计算一批命中结果的最高数据级别
"""

from __future__ import annotations

from typing import Any, Iterable

from core.models.document import ChunkMetadata


CLASSIFICATION_ORDER = {
    "public": 0,
    "internal": 1,
    "sensitive": 2,
    "restricted": 3,
}
ACL_METADATA_KEYS = ("allowed_users", "allowed_departments", "allowed_roles", "project_ids")


def _normalize_text(value: object) -> str | None:
    """把字符串标准化为去空白后的文本。"""

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text


def _coerce_to_list(value: object) -> list[str]:
    """把 metadata 中的字符串或列表统一收敛为字符串列表。"""

    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = _normalize_text(item)
            if text:
                out.append(text)
        return out
    text = _normalize_text(value)
    return [text] if text else []


def normalize_classification(value: object, default: str = "internal") -> str:
    """把原始分级值规整到支持的分类枚举内。"""

    text = _normalize_text(value)
    if text is None:
        return default
    lowered = text.lower()
    if lowered in CLASSIFICATION_ORDER:
        return lowered
    return default


def build_access_filters(user_context: dict[str, Any] | None, settings: Any) -> dict[str, Any]:
    """从用户上下文生成后续 retrieval 可复用的访问过滤条件。

    这里先把请求上下文规整成统一结构，是为了避免：
    - chat route 一套字段名
    - retrieval 一套字段名
    - ACL 判断再来一套字段名

    先收敛成稳定 access filters，后续链路就可以复用同一个输入。
    """

    ctx = user_context or {}
    filters: dict[str, Any] = {}

    user_id = _normalize_text(ctx.get("user_id"))
    department = _normalize_text(ctx.get("department"))
    role = _normalize_text(ctx.get("role"))
    project_ids = _coerce_to_list(ctx.get("project_ids"))
    clearance_level = normalize_classification(
        ctx.get("clearance_level"),
        getattr(settings, "default_data_classification", "internal"),
    )

    if user_id:
        filters["user_id"] = user_id
    if department:
        filters["department"] = department
    if role:
        filters["role"] = role
    if project_ids:
        filters["project_ids"] = project_ids
    filters["clearance_level"] = clearance_level
    return filters


def build_retrieval_acl_filters(access_filters: dict[str, Any] | None) -> dict[str, Any]:
    """把用户上下文过滤条件映射成检索层可下传的 ACL metadata 过滤条件。

    这一步的本质是把“用户是谁”翻译成“允许搜哪些 chunk”：
    - 用户 id -> `allowed_users`
    - 部门 -> `allowed_departments`
    - 角色 -> `allowed_roles`
    - 项目 -> `project_ids`
    """

    if not isinstance(access_filters, dict):
        return {}

    filters: dict[str, Any] = {}
    user_id = _normalize_text(access_filters.get("user_id"))
    department = _normalize_text(access_filters.get("department"))
    role = _normalize_text(access_filters.get("role"))
    project_ids = _coerce_to_list(access_filters.get("project_ids"))

    if user_id:
        filters["allowed_users"] = user_id
    if department:
        filters["allowed_departments"] = department
    if role:
        filters["allowed_roles"] = role
    if project_ids:
        filters["project_ids"] = project_ids
    return filters


def get_chunk_classification(metadata: ChunkMetadata, settings: Any) -> str:
    """读取 chunk 的数据分级，没有显式配置时退回默认分级。

    数据分级是 ACL 之外的另一道门：
    即使 ACL 命中，如果用户密级不够，也不应该放行。
    """

    return normalize_classification(
        metadata.extra.get("data_classification"),
        getattr(settings, "default_data_classification", "internal"),
    )


def _has_acl_metadata(metadata: ChunkMetadata) -> bool:
    """判断 chunk 是否携带了至少一种 ACL 元数据。"""

    return any(_coerce_to_list(metadata.extra.get(key)) for key in ACL_METADATA_KEYS)


def is_chunk_accessible(
    metadata: ChunkMetadata,
    user_context: dict[str, Any] | None,
    settings: Any,
) -> bool:
    """判断当前用户是否有权访问某个 chunk。

    判断顺序体现了企业场景下的真实优先级：
    1. 先看数据分级是否允许访问
    2. 再看 ACL 是否命中
    3. 对高敏 chunk，在 strict mode 下要求必须显式带 ACL

    这里强调“检索前/检索中拦截”，而不是生成后裁剪。
    """

    ctx = user_context or {}
    chunk_classification = get_chunk_classification(metadata, settings)
    user_clearance = normalize_classification(
        ctx.get("clearance_level"),
        getattr(settings, "default_data_classification", "internal"),
    )

    if getattr(settings, "enable_data_classification", False):
        if CLASSIFICATION_ORDER[user_clearance] < CLASSIFICATION_ORDER[chunk_classification]:
            return False

    if not getattr(settings, "enable_acl", False):
        return True

    if (
        getattr(settings, "acl_strict_mode", False)
        and chunk_classification in {"sensitive", "restricted"}
        and not _has_acl_metadata(metadata)
    ):
        return False

    allowed_users = set(_coerce_to_list(metadata.extra.get("allowed_users")))
    allowed_departments = set(_coerce_to_list(metadata.extra.get("allowed_departments")))
    allowed_roles = set(_coerce_to_list(metadata.extra.get("allowed_roles")))
    allowed_projects = set(_coerce_to_list(metadata.extra.get("project_ids")))

    user_id = _normalize_text(ctx.get("user_id"))
    department = _normalize_text(ctx.get("department"))
    role = _normalize_text(ctx.get("role"))
    project_ids = set(_coerce_to_list(ctx.get("project_ids")))

    if allowed_users and user_id not in allowed_users:
        return False
    if allowed_departments and department not in allowed_departments:
        return False
    if allowed_roles and role not in allowed_roles:
        return False
    if allowed_projects and not (project_ids & allowed_projects):
        return False
    return True


def resolve_model_route(
    *,
    settings: Any,
    data_classification: str,
    allow_external_llm: bool | None = None,
) -> str:
    """根据数据分级和请求策略推导模型路由标签。

    这一步不直接调用模型，只负责输出一个清晰的路由结论，例如：
    - `local_only`
    - `local_preferred`
    - `external_allowed`

    这样生成层可以只消费路由结果，而不需要重复理解 ACL/分级规则。
    """

    if not getattr(settings, "enable_model_routing", False):
        return "default"

    classification = normalize_classification(
        data_classification,
        getattr(settings, "default_data_classification", "internal"),
    )
    local_only = {
        normalize_classification(item, getattr(settings, "default_data_classification", "internal"))
        for item in getattr(settings, "local_only_classifications", [])
    }

    if allow_external_llm is False:
        return "local_only"
    if classification in local_only:
        return "local_only"
    if classification == "sensitive":
        if allow_external_llm is True or getattr(settings, "allow_external_llm_for_sensitive", False):
            return "external_allowed"
        return "local_preferred"
    return "external_allowed"


def resolve_data_classification(
    items: Iterable[ChunkMetadata | dict[str, Any]],
    *,
    default: str = "internal",
) -> str:
    """返回一批命中结果中的最高数据分级。

    一次问答往往会命中多个 chunk。
    模型路由和审计不能只看其中最宽松的一条，而要看“这批证据里的最高敏感级别”。
    """

    highest = normalize_classification(default, default)
    for item in items:
        if isinstance(item, ChunkMetadata):
            current = normalize_classification(item.extra.get("data_classification"), highest)
        else:
            metadata = item.get("metadata") if isinstance(item, dict) else None
            if isinstance(metadata, ChunkMetadata):
                current = normalize_classification(metadata.extra.get("data_classification"), highest)
            elif isinstance(metadata, dict):
                extra = metadata.get("extra") or {}
                current = normalize_classification(extra.get("data_classification"), highest)
            else:
                current = normalize_classification(item.get("data_classification"), highest)
        if CLASSIFICATION_ORDER[current] > CLASSIFICATION_ORDER[highest]:
            highest = current
    return highest
