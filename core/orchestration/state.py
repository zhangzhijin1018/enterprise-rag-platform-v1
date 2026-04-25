"""LangGraph 状态定义模块。

这里定义的 `RAGState` 可以理解成“问答主链路的共享状态总表”：

- 每个节点只关心自己要读写的那部分字段
- 所有中间产物都往同一个状态对象里汇总
- API 层最终也从这里取最终答案、引用、拒答原因和审计信息

为什么用 `TypedDict(total=False)` 而不是要求所有字段都必填：
- LangGraph 每个节点通常只返回“状态增量”
- 不是所有字段在一开始就存在
- 用可选字段更符合编排式流水线的真实形态
"""

from __future__ import annotations

from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    """RAG 主链路共享状态。

    阅读这个结构时，最容易理解的方式不是逐字段死记，而是按阶段看：

    1. 输入阶段：问题、历史、top_k、用户上下文
    2. 分析阶段：query understanding、澄清、上下文补全
    3. 检索阶段：query rewrite、召回、融合、重排
    4. 安全阶段：ACL、分级、风控、模型路由
    5. 生成阶段：答案、引用、拒答、grounding 校验

    换句话说，`RAGState` 不是一个普通 DTO，
    而是整条 LangGraph 执行过程中的“过程快照”。
    """

    # 原始输入与运行时控制参数。
    question: str  # 用户当前问题
    conversation_id: str | None  # 会话 id
    history_messages: list[dict[str, Any]]  # 对话历史
    top_k_sparse: int | None  # 稀疏检索 top_k
    top_k_dense: int | None  # 稠密检索 top_k
    rerank_top_n: int | None  # 最终重排保留数量

    # 查询分析、澄清与上下文补全阶段产物。
    strategy_signals: dict[str, Any]  # 查询理解后的策略信号总表；决定走什么检索策略、是否需要澄清等
    need_clarify: bool  # 是否需要先追问
    missing_slots: list[str]  # 当前缺失的关键槽位
    clarify_question: str  # 追问内容
    clarify_reason: str  # 为什么要追问
    resolved_query: str  # 多轮补全后的完整查询；用于把“今天谁值班”补成带部门/班次的完整问题
    structured_filters: dict[str, Any]  # 显式结构化过滤条件；例如 department/shift/time
    metadata_intent: dict[str, Any]  # 面向检索的 metadata 意图；例如 business_domain/process_stage
    analysis_confidence: float  # query understanding 置信度
    analysis_source: str  # heuristic / llm_enhanced / *_guardrail
    analysis_reason: str  # 置信度与路由判定原因

    # 查询规划阶段产物。
    rewritten_query: str  # 轻量改写后的主查询；尽量更适合直接检索
    keyword_queries: list[str]  # 适合 sparse 的关键词查询；偏词面约束
    multi_queries: list[str]  # 拆出来的子查询；用于比较、流程分步等复杂问题
    hyde_query: str  # HyDE 用的假设答案查询；在需要时给 dense 一个更“像答案”的查询表达
    query_plan_summary: str  # 查询规划摘要

    # 检索与重排阶段产物。
    sparse_hits: list[dict[str, Any]]  # BM25 / sparse 命中
    dense_hits: list[dict[str, Any]]  # dense / 向量检索命中
    fused_hits: list[dict[str, Any]]  # 融合后的候选；是“是否值得继续 rerank”的第一道结果
    reranked_hits: list[dict[str, Any]]  # 精排后的最终上下文候选；通常就是 generation 的主要证据来源

    # 企业安全与访问控制上下文。
    user_context: dict[str, Any]  # 当前用户上下文
    access_filters: dict[str, Any]  # ACL / ABAC 检索过滤条件；优先在检索前/检索中生效
    data_classification: str  # 命中文档的最高数据分级；用于模型路由、审计和拒答判断
    model_route: str  # external_allowed / local_only 等模型路由标签
    answer_mode: str  # grounded_answer / refusal / local_grounded_answer 等；帮助 API/前端统一识别回答类型
    audit_id: str  # 审计追踪 id
    risk_level: str  # 风险等级
    risk_action: str  # 风控动作
    risk_reason: str  # 风控原因
    risk_require_alert: bool  # 是否需要触发告警
    conflict_detected: bool  # 是否检测到多文档冲突
    conflict_summary: str  # 冲突摘要

    # 生成与最终校验阶段产物。
    answer: str  # 最终答案
    confidence: float  # 当前回答置信度
    reasoning_summary: str  # 简要推理摘要
    citations: list[dict[str, Any]]  # 引用列表；尽量返回可追溯证据而不是裸答案
    fast_path_source: str  # 若命中 fast path，记录来源；如 redis_answer_cache / mysql_faq
    refusal: bool  # 是否拒答
    refusal_reason: str  # 拒答原因

    grounding_ok: bool  # grounded validation 是否通过
    errors: list[str]  # 链路中的错误信息
