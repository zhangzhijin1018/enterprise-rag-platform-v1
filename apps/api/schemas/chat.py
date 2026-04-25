"""问答接口的数据模型模块。

这一层是“对外契约”，重点不是内部怎么实现，而是告诉调用方：
- 该传什么
- 能收到什么
- 哪些字段和企业安全、引用、拒答相关

所以这里的注释重点应当始终偏“调用语义”，而不是内部实现细节。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HistoryMessageSchema(BaseModel):
    """单轮历史消息。

    历史消息不会直接拼进最终回答，
    而是先用于上下文补全、指代消解和 query understanding。
    """

    role: str = Field(default="user", description="如 user / assistant")
    content: str = Field(description="该轮消息内容")


class ChatRequest(BaseModel):
    """问答请求体。

    可以把字段分成三类来理解：
    - 问答控制：`question / top_k / stream / history_messages`
    - 企业上下文：`user_id / department / role / project_ids / clearance_level`
    - 策略偏好：`query_scene / require_citations / allow_external_llm`
    """

    question: str = Field(description="用户当前问题")
    conversation_id: str | None = Field(default=None, description="会话 id")
    history_messages: list[HistoryMessageSchema] = Field(default_factory=list, description="历史消息")
    top_k: int = Field(default=8, ge=1, le=64, description="覆盖本轮 rerank top_n")
    stream: bool = Field(default=False, description="是否使用流式返回")
    user_id: str | None = Field(default=None, description="用户唯一标识")
    username: str | None = Field(default=None, description="用户名")
    department: str | None = Field(default=None, description="所属部门")
    role: str | None = Field(default=None, description="角色")
    project_ids: list[str] = Field(default_factory=list, description="关联项目 id 列表")
    clearance_level: str | None = Field(default=None, description="用户密级/权限级别")
    query_scene: str | None = Field(default=None, description="调用方显式指定的问题场景")
    require_citations: bool = Field(default=True, description="是否要求返回引用")
    allow_external_llm: bool | None = Field(default=None, description="是否允许外部模型参与")
    session_metadata: dict[str, Any] = Field(default_factory=dict, description="会话扩展 metadata")


class CitationSchema(BaseModel):
    """对外返回的引用结构。

    它的目标不是“把内部 metadata 原样透出”，
    而是优先返回真正对用户有用、能解释答案来源的字段。
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
    version: str | None = Field(default=None, description="文档版本")
    effective_date: str | None = Field(default=None, description="生效日期")
    authority_level: str | None = Field(default=None, description="权威级别")
    source_system: str | None = Field(default=None, description="来源系统")
    business_domain: str | None = Field(default=None, description="业务域")
    process_stage: str | None = Field(default=None, description="流程阶段")
    section_path: str | None = Field(default=None, description="章节路径")
    matched_routes: list[str] = Field(default_factory=list, description="命中的 query route")
    retrieval_score: float | None = Field(default=None, description="最终召回分")
    semantic_score: float | None = Field(default=None, description="语义/重排分")
    governance_rank_score: float | None = Field(default=None, description="治理排序后的分数")
    selection_reason: str | None = Field(default=None, description="为什么这条证据被选中")


class RetrievedChunkSchema(BaseModel):
    """对外暴露的检索片段结构。

    这个结构主要用于调试、解释性展示和前端证据面板，
    不是给模型再消费的内部对象。
    """

    chunk_id: str = Field(description="chunk id")
    score: float = Field(description="检索或重排分数")
    content: str = Field(description="chunk 内容")
    metadata: dict[str, Any] = Field(description="chunk metadata")


class ChatResponse(BaseModel):
    """问答响应体。

    这里把普通问答字段和企业治理字段统一放在一个响应里，
    这样前端拿到一份 JSON 就能同时判断：
    - 回答内容
    - 是否拒答
    - 是否有引用
    - 命中了什么安全/分析/冲突信息
    """

    answer: str = Field(description="最终答案")
    confidence: float = Field(description="回答置信度")
    fast_path_source: str | None = Field(default=None, description="若命中 fast path，记录来源")
    citations: list[CitationSchema] = Field(description="引用列表")
    retrieved_chunks: list[RetrievedChunkSchema] = Field(description="召回到的 chunk 列表")
    refusal: bool = Field(default=False, description="是否拒答")
    refusal_reason: str | None = Field(default=None, description="拒答原因")
    answer_mode: str | None = Field(default=None, description="回答模式")
    data_classification: str | None = Field(default=None, description="命中内容的数据分级")
    model_route: str | None = Field(default=None, description="模型路由结果")
    analysis_confidence: float | None = Field(default=None, description="query understanding 置信度")
    analysis_source: str | None = Field(default=None, description="query understanding 来源")
    analysis_reason: str | None = Field(default=None, description="query understanding 说明")
    conflict_detected: bool = Field(default=False, description="是否检测到证据冲突")
    conflict_summary: str | None = Field(default=None, description="冲突摘要")
    trace_id: str | None = Field(default=None, description="请求链路追踪 id")
    audit_id: str | None = Field(default=None, description="审计追踪 id")
