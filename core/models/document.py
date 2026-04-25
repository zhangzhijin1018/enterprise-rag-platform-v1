"""文档与 chunk 领域模型模块。

这些模型是 ingestion、retrieval、generation 三层之间的“公共语言”：

- `Document`：表示 parser 刚解析完、还没切块的整篇文档
- `TextChunk`：表示已经切好的、可进入索引或回传引用的文本片段
- `ChunkMetadata`：表示 chunk 的来源、定位信息、治理字段和扩展语义

为什么这一层很关键：

1. parser、chunker、retriever、citation formatter 都依赖同一套结构
2. 字段语义一旦混乱，后面 ACL、过滤、引用、审计都会跟着混乱
3. 所以这里要尽量把“什么是文档级字段，什么是 chunk 级字段”分清楚
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

PARENT_CHUNK_LEVEL = "parent"
CHILD_CHUNK_LEVEL = "child"
# 企业文档常见的标量 metadata 字段。
# 这些字段大多会进入：
# 1. ingestion 清洗与抽取
# 2. retrieval 过滤与排序
# 3. citation 对外展示
ENTERPRISE_SCALAR_METADATA_KEYS = (
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
)
# 企业文档常见的列表型 metadata 字段。
# 这类字段通常和 ACL / 项目归属等多值属性相关。
ENTERPRISE_LIST_METADATA_KEYS = (
    "allowed_users",
    "allowed_roles",
    "allowed_departments",
    "project_ids",
)
ENTERPRISE_METADATA_KEYS = ENTERPRISE_SCALAR_METADATA_KEYS + ENTERPRISE_LIST_METADATA_KEYS
# chunk 局部语义字段。
# 这类字段不一定是整篇文档的属性，而是当前 chunk 的局部语义标签。
CHUNK_SEMANTIC_SCALAR_KEYS = (
    "section_path",
    "section_level",
    "section_type",
    "chunk_summary",
    "contains_table",
    "contains_steps",
    "contains_contact",
    "contains_version_signal",
    "contains_risk_signal",
)
CHUNK_SEMANTIC_LIST_KEYS = ("topic_keywords",)


def normalize_metadata_scalar(value: Any) -> str | None:
    """把元数据值规整成去空白字符串。

    目的不是“美化数据”，而是减少 retrieval / filtering 时的类型分叉：
    后续很多链路默认把 metadata 当成稳定字符串处理。
    """

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_metadata_list(value: Any) -> list[str]:
    """把元数据值规整成字符串列表。

    ACL、项目归属、适用部门这类字段天然是多值字段。
    统一收敛成 `list[str]` 后，后续 metadata filter 的行为会更稳定。
    """

    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = normalize_metadata_scalar(item)
            if text:
                out.append(text)
        return out
    text = normalize_metadata_scalar(value)
    return [text] if text else []


def normalize_enterprise_metadata_value(key: str, value: Any) -> str | list[str] | None:
    """按 metadata key 的类型收敛成稳定值。

    这里相当于“企业 metadata 的轻量 schema 守门员”：
    - 标量字段统一成 `str`
    - 列表字段统一成 `list[str]`
    - 空值统一收敛成 `None`
    """

    if key in ENTERPRISE_LIST_METADATA_KEYS:
        values = normalize_metadata_list(value)
        return values or None
    return normalize_metadata_scalar(value)


class ChunkMetadata(BaseModel):
    """chunk 的检索与引用元数据。

    `extra` 字段保留了扩展空间，用来承载不适合立刻提升为一级字段、
    但又在特定链路里非常关键的附加信息。

    在第三轮增强里，`extra` 里会重点放这几类信息：

    - `chunk_level`：当前 chunk 是 `parent` 还是 `child`
    - `parent_chunk_id`：child chunk 对应的 parent chunk

    为什么不直接把这些字段一上来就提升为一级字段：

    1. 这样能保持当前 API schema 和磁盘结构的兼容性更好；
    2. 未来如果还要加 `tenant_id`、`acl_tag`、`table_id` 等扩展字段，
       仍然可以沿用同一套扩展槽位。

    可以把它理解成两层：
    - 一级字段：所有 chunk 都稳定存在、而且高频使用的定位信息
    - `extra`：和治理、检索增强、解释性相关，但仍在逐步演进的扩展信息
    """

    doc_id: str = Field(description="文档唯一标识")
    chunk_id: str = Field(description="chunk 唯一标识")
    source: str = Field(description="原始来源，例如文件名、URL 或外部系统路径")
    title: str = Field(default="", description="文档标题")
    page: int | None = Field(default=None, description="页码；非分页文档可为空")
    section: str | None = Field(default=None, description="当前 chunk 所在章节标题")
    extra: dict[str, Any] = Field(default_factory=dict, description="扩展 metadata 槽位")

    @property
    def chunk_level(self) -> str:
        """返回当前 chunk 的层级。

        兼容策略：
        - 老索引里没有 `chunk_level` 字段时，默认把它当作 `child`
        - 这样旧数据不会因为升级分层检索而全部失效
        """

        raw = self.extra.get("chunk_level")
        if raw == PARENT_CHUNK_LEVEL:
            return PARENT_CHUNK_LEVEL
        return CHILD_CHUNK_LEVEL

    @property
    def parent_chunk_id(self) -> str | None:
        """返回当前 chunk 对应的父 chunk id。"""

        raw = self.extra.get("parent_chunk_id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return None

    @property
    def is_parent(self) -> bool:
        """是否是 parent chunk。"""

        return self.chunk_level == PARENT_CHUNK_LEVEL

    @property
    def is_child(self) -> bool:
        """是否是 child chunk。"""

        return self.chunk_level == CHILD_CHUNK_LEVEL

    @property
    def searchable(self) -> bool:
        """当前 chunk 是否应该直接进入召回索引。

        当前策略：
        - parent chunk 主要用于给 rerank / generation 提供更完整上下文
        - child chunk 主要用于召回

        所以默认只有 child chunk 参与 BM25 / Dense 检索。
        """

        return self.is_child

    def extra_text(self, key: str) -> str | None:
        """读取扩展元数据中的文本值。"""

        return normalize_metadata_scalar(self.extra.get(key))

    def extra_list(self, key: str) -> list[str]:
        """读取扩展元数据中的列表值。"""

        return normalize_metadata_list(self.extra.get(key))


class Document(BaseModel):
    """解析后的标准文档对象。

    `Document` 是 parser 和 chunker 之间的交接格式。

    它强调“整篇文档视角”：
    - `content` 还是完整正文
    - `metadata` 还是文档级属性
    - 还没有进入 parent / child chunk 的粒度

    这也是为什么：
    - `doc_id` 放在这里
    - `doc_type / classification / authority` 这类字段先挂在这里
    - 后续切块时，再把需要的字段下沉到每个 chunk 的 `metadata.extra`
    """

    doc_id: str = Field(description="文档唯一标识")
    source: str = Field(description="文档来源")
    title: str = Field(default="", description="文档标题")
    content: str = Field(description="解析后的完整正文")
    mime_type: str | None = Field(default=None, description="原始 MIME 类型")
    metadata: dict[str, Any] = Field(default_factory=dict, description="文档级原始 metadata")


class TextChunk(BaseModel):
    """可进入索引或用于引用展示的最小文本单元。

    可以把它理解成“RAG 真正工作的基本粒度”：

    - 检索器按 chunk 召回
    - reranker 按 chunk 打分
    - generation 往往拿一组 chunk 来拼上下文
    - citation 最终也通常回指到 chunk

    所以 `TextChunk = 正文内容 + 可追踪 metadata`，两者缺一不可。
    """

    content: str = Field(description="chunk 正文")
    metadata: ChunkMetadata = Field(description="chunk 对应的检索与来源 metadata")

    @property
    def is_parent(self) -> bool:
        """代理到 metadata，便于上层代码少写一层访问。"""

        return self.metadata.is_parent

    @property
    def is_child(self) -> bool:
        """代理到 metadata，便于上层代码少写一层访问。"""

        return self.metadata.is_child

    @property
    def searchable(self) -> bool:
        """代理到 metadata，便于检索器统一过滤可搜索 chunk。"""

        return self.metadata.searchable
