"""问答接口路由模块。

这个文件是前端调用最频繁的入口：
- 非流式模式走完整 LangGraph 问答链路后一次性返回；
- 流式模式先跑检索，再把 LLM token 通过 NDJSON 持续推给前端。
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from apps.api.dependencies.common import get_runtime_dep
from apps.api.schemas.chat import ChatRequest, ChatResponse, CitationSchema, RetrievedChunkSchema
from core.generation.answer_builder import parse_llm_grounded_output
from core.generation.context_format import format_context_blocks
from core.generation.egress_policy import prepare_contexts_for_generation
from core.generation.llm_client import LLMClient
from core.generation.local_executor import build_local_grounded_output
from core.generation.prompts.templates import GROUNDED_ANSWER_SYSTEM
from core.observability import get_logger, update_request_log_context
from core.observability.audit import log_alert_event, log_audit_event
from core.orchestration.graph import run_rag_async
from core.orchestration.retrieval_pipeline import run_retrieval_only
from core.retrieval.access_control import build_access_filters
from core.retrieval.schemas import RetrievedChunk
from core.security.ml_risk_provider import (
    build_request_risk_feature_bundle,
    safe_predict_ml_risk_hint,
)
from core.security.risk_engine import (
    RuleBasedRiskEngine,
    build_risk_context,
    decision_to_state_update,
    safe_evaluate_risk,
)
from core.services.runtime import RAGRuntime

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


def _chunks_from_state(state: dict) -> list[RetrievedChunkSchema]:
    """把图状态里的 chunk 字典列表转成 API schema。

    LangGraph 内部状态里常常是普通字典，
    这里统一在出接口前收敛成响应模型，避免前端直接依赖内部状态结构。
    """

    raw = state.get("reranked_hits") or []
    out: list[RetrievedChunkSchema] = []
    for x in raw:
        out.append(
            RetrievedChunkSchema(
                chunk_id=x["chunk_id"],
                score=float(x.get("score", 0.0)),
                content=x.get("content", ""),
                metadata=(x.get("metadata") or {}),
            )
        )
    return out


def _citations_from_state(state: dict) -> list[CitationSchema]:
    """把图状态里的引用列表转成对外响应模型。

    这样做的价值在于：
    - API 层统一校验字段完整性
    - 内部 citation 结构就算以后演进，对外契约也更稳定
    """

    return [CitationSchema.model_validate(c) for c in (state.get("citations") or [])]


def _build_user_context(body: ChatRequest) -> dict[str, Any]:
    """从请求体里抽取企业用户上下文。

    这里刻意不把整个 `ChatRequest` 直接往下透传，
    而是先收敛成一份更稳定的 `user_context`，
    方便 ACL、风控、模型路由等链路共享。
    """

    return {
        "user_id": body.user_id,
        "username": body.username,
        "department": body.department,
        "role": body.role,
        "project_ids": body.project_ids,
        "clearance_level": body.clearance_level,
        "query_scene": body.query_scene,
        "require_citations": body.require_citations,
        "allow_external_llm": body.allow_external_llm,
        "session_metadata": body.session_metadata,
    }


def _ml_risk_state_update(result: Any) -> dict[str, Any]:
    """把 ML 风控 hint 收敛到状态字段。

    当前先只作为内部状态和审计补充，不对外扩展 API schema。
    """

    if result is None:
        return {}
    return {
        "ml_risk_hint": result.risk_level_hint,
        "ml_risk_confidence": result.confidence,
        "ml_risk_provider": result.provider,
        "ml_risk_fallback": result.fallback,
    }


def _has_enterprise_context(body: ChatRequest) -> bool:
    """判断本次请求是否携带了企业安全上下文。

    这个判断主要用于 fast path 决策：
    当 FAQ / cache 还没有 ACL 化时，带企业安全上下文的请求要更谨慎。
    """

    return any(
        [
            body.user_id,
            body.username,
            body.department,
            body.role,
            body.project_ids,
            body.clearance_level,
            body.query_scene,
            body.allow_external_llm is not None,
            body.session_metadata,
        ]
    )


def _should_disable_fast_path(runtime: RAGRuntime, body: ChatRequest) -> bool:
    """在当前 FAQ / cache 尚未 ACL 化前，带安全上下文的请求绕过 fast path。

    这是一个典型的企业化保护策略：
    - 无上下文的通用问题，可以优先走快路径
    - 带权限/密级/项目上下文的问题，宁可慢一点，也要走可控主链路
    """

    settings = runtime.settings
    security_enabled = bool(
        getattr(settings, "enable_acl", False)
        or getattr(settings, "enable_data_classification", False)
        or getattr(settings, "enable_model_routing", False)
    )
    return security_enabled and _has_enterprise_context(body)


def _response_security_fields(
    state: dict[str, Any],
    *,
    trace_id: str | None = None,
    audit_id: str,
    answer_mode: str | None = None,
    refusal: bool | None = None,
    refusal_reason: str | None = None,
) -> dict[str, Any]:
    """统一收敛响应里的企业安全字段。

    这样非流式、流式、拒答、fast path 都能复用同一套响应字段口径，
    前端也不需要为不同返回路径写多套解析逻辑。
    """

    resolved_refusal = bool(state.get("refusal")) if refusal is None else refusal
    resolved_answer_mode = answer_mode or state.get("answer_mode")
    if not resolved_answer_mode:
        if state.get("fast_path_source"):
            resolved_answer_mode = "fast_path"
        elif resolved_refusal:
            resolved_answer_mode = "refusal"
        else:
            resolved_answer_mode = "grounded_answer"
    return {
        "refusal": resolved_refusal,
        "refusal_reason": refusal_reason if refusal_reason is not None else state.get("refusal_reason"),
        "answer_mode": resolved_answer_mode,
        "data_classification": state.get("data_classification"),
        "model_route": state.get("model_route"),
        "analysis_confidence": state.get("analysis_confidence"),
        "analysis_source": state.get("analysis_source"),
        "analysis_reason": state.get("analysis_reason"),
        "conflict_detected": bool(state.get("conflict_detected")),
        "conflict_summary": state.get("conflict_summary"),
        "trace_id": str(state.get("trace_id") or trace_id or "") or None,
        "audit_id": str(state.get("audit_id") or audit_id),
    }


def _normalize_state_defaults(state: dict[str, Any], *, audit_id: str) -> dict[str, Any]:
    """确保后续响应构造能拿到稳定的安全字段。

    图状态是按需逐步补字段的，
    这里在出接口前补默认值，可以减少后面构造响应时的分支判断。
    """

    out = dict(state)
    out.setdefault("audit_id", audit_id)
    out.setdefault("refusal", False)
    out.setdefault("conflict_detected", False)
    out.setdefault("analysis_source", None)
    return out


def _emit_security_alert(
    *,
    runtime: RAGRuntime,
    audit_id: str,
    question: str,
    user_context: dict[str, Any],
    state: dict[str, Any],
    stream: bool,
) -> None:
    """统一触发安全告警。

    抽成公共函数是为了让：
    - 流式/非流式
    - 正常回答/拒答
    - 请求级风控/生成级风控
    都走同一条告警出口。
    """

    log_alert_event(
        audit_id=audit_id,
        question=question,
        user_context=user_context,
        settings=runtime.settings,
        state=state,
        extra={"stream": stream},
    )


def _log_chain_event(event: str, message: str, **extra: Any) -> None:
    """记录问答主链路关键步骤日志。"""

    logger.info(message, extra={"event": event, **extra})


async def _run_graph(
    runtime: RAGRuntime,
    body: ChatRequest,
    *,
    user_context: dict[str, Any],
    access_filters: dict[str, Any],
    risk_state: dict[str, Any],
    audit_id: str,
) -> dict:
    """执行完整 RAG 图，并根据前端 `top_k` 派生召回参数。

    设计原因：
    - 前端只感知一个 `top_k`，避免暴露太多底层参数。
    - 内部会把召回数量放大，再交给 rerank 收敛到最终的 top_n，
      这样能兼顾召回率和最终答案质量。

    这里相当于 API 层到编排层的一层“参数翻译”：
    对外接口尽量简单，对内仍保留足够的检索控制能力。
    """

    sk = min(body.top_k * 2, runtime.settings.bm25_top_k + 10)
    dk = min(body.top_k * 2, runtime.settings.dense_top_k + 10)
    return await run_rag_async(
        runtime,
        question=body.question,
        conversation_id=body.conversation_id,
        history_messages=[item.model_dump(mode="json") for item in body.history_messages],
        top_k_sparse=sk,
        top_k_dense=dk,
        rerank_top_n=body.top_k,
        user_context=user_context,
        access_filters=access_filters,
        risk_state=risk_state,
        audit_id=audit_id,
        disable_fast_path=_should_disable_fast_path(runtime, body),
    )


def _build_risk_refusal_state(
    *,
    audit_id: str,
    risk_state: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """构造入口级风控拒答状态。

    入口风控的特点是：
    - 图还没执行
    - 检索也还没开始
    - 但请求本身已经不该继续处理

    所以这里单独构造一份和主链路兼容的 refusal state，便于统一响应。
    """

    return {
        "answer": "当前请求命中了企业风控策略，已拒绝处理。",
        "confidence": 0.0,
        "reasoning_summary": "请求入口命中了高风险规则。",
        "citations": [],
        "reranked_hits": [],
        "retrieved_chunks": [],
        "refusal": True,
        "refusal_reason": reason,
        "answer_mode": "refusal",
        "audit_id": audit_id,
        **risk_state,
    }


@router.post("")
async def chat(
    request: Request,
    body: ChatRequest,
    runtime: RAGRuntime = Depends(get_runtime_dep),
):
    """统一处理流式与非流式问答请求。

    这个路由本质上做四件事：
    1. 收请求，构造用户上下文、访问过滤和审计 id
    2. 先过请求级风控，必要时提前拒答
    3. 根据 `stream` 选择“检索优先流式”或“完整图非流式”
    4. 把内部状态统一收敛成对外响应
    """

    user_context = _build_user_context(body)
    access_filters = build_access_filters(user_context, runtime.settings)
    audit_id = uuid4().hex
    trace_id = getattr(request.state, "trace_id", "")
    update_request_log_context(
        trace_id=trace_id,
        audit_id=audit_id,
        user_id=user_context.get("user_id"),
        department=user_context.get("department"),
        role=user_context.get("role"),
        project_ids=user_context.get("project_ids") or [],
        event="request_received",
    )
    _log_chain_event(
        "request_received",
        "chat request received",
        stream=body.stream,
        top_k=body.top_k,
    )
    request_risk_context = build_risk_context(
        stage="request",
        question=body.question,
        audit_id=audit_id,
        user_context=user_context,
        state={"access_filters": access_filters, "trace_id": trace_id},
    )
    ml_risk_result = safe_predict_ml_risk_hint(
        getattr(runtime, "ml_risk_provider", None),
        context=request_risk_context,
        feature_bundle=build_request_risk_feature_bundle(question=body.question, user_context=user_context),
        settings=runtime.settings,
    )
    if ml_risk_result.risk_level_hint:
        request_risk_context.risk_level_hint = ml_risk_result.risk_level_hint
    risk_engine = getattr(runtime, "risk_engine", RuleBasedRiskEngine(runtime.settings))
    request_risk_decision = safe_evaluate_risk(risk_engine, request_risk_context, runtime.settings)
    risk_state = {
        **decision_to_state_update(request_risk_decision),
        **_ml_risk_state_update(ml_risk_result),
    }
    log_audit_event(
        stage="request_received",
        audit_id=audit_id,
        question=body.question,
        user_context=user_context,
        settings=runtime.settings,
        state=risk_state,
        extra={
            "stream": body.stream,
            "top_k": body.top_k,
            "ml_risk_hint": ml_risk_result.risk_level_hint,
            "ml_risk_provider": ml_risk_result.provider,
            "ml_risk_fallback": ml_risk_result.fallback,
        },
    )

    if not request_risk_decision.allow:
        # 请求级风控优先级最高。
        # 这类场景下无需进入 graph，直接返回拒答更稳，也更省成本。
        _log_chain_event(
            "request_denied",
            "request denied by risk engine before graph execution",
            risk_reason=request_risk_decision.reason or "risk_engine_denied",
        )
        refusal_state = _build_risk_refusal_state(
            audit_id=audit_id,
            risk_state=risk_state,
            reason=request_risk_decision.reason or "risk_engine_denied",
        )
        refusal_state["trace_id"] = trace_id
        log_audit_event(
            stage="response_sent",
            audit_id=audit_id,
            question=body.question,
            user_context=user_context,
            settings=runtime.settings,
            state=refusal_state,
            output=refusal_state["answer"],
            extra={"stream": body.stream, "risk_stage": "request"},
        )
        _emit_security_alert(
            runtime=runtime,
            audit_id=audit_id,
            question=body.question,
            user_context=user_context,
            state=refusal_state,
            stream=body.stream,
        )
        if body.stream:
            async def gen_denied():
                # 即使是入口级拒答，流式协议也保持 `meta -> token -> final` 三段式，
                # 这样前端不需要为拒答单独维护另一套流协议。
                meta = {
                    "citations": [],
                    "retrieved_chunks": [],
                    "confidence": 0.0,
                    "refusal": True,
                    **_response_security_fields(
                        refusal_state,
                        audit_id=audit_id,
                        answer_mode="refusal",
                        refusal=True,
                        refusal_reason=refusal_state["refusal_reason"],
                    ),
                }
                yield json.dumps({"type": "meta", "data": meta}, ensure_ascii=False) + "\n"
                yield json.dumps(
                    {"type": "token", "data": refusal_state["answer"]},
                    ensure_ascii=False,
                ) + "\n"
                yield json.dumps(
                    {
                        "type": "final",
                        "data": {
                            "answer": refusal_state["answer"],
                            "confidence": 0.0,
                            "citations": [],
                            **_response_security_fields(
                                refusal_state,
                                audit_id=audit_id,
                                answer_mode="refusal",
                                refusal=True,
                                refusal_reason=refusal_state["refusal_reason"],
                            ),
                        },
                    },
                    ensure_ascii=False,
                ) + "\n"
            return StreamingResponse(gen_denied(), media_type="application/x-ndjson")

        resp = ChatResponse(
            answer=refusal_state["answer"],
            confidence=0.0,
            fast_path_source=None,
            citations=[],
            retrieved_chunks=[],
            **_response_security_fields(
                refusal_state,
                audit_id=audit_id,
                answer_mode="refusal",
                refusal=True,
                refusal_reason=refusal_state["refusal_reason"],
            ),
        )
        _log_chain_event("response_sent", "request denied response sent", refusal=True)
        return JSONResponse(resp.model_dump())

    if body.stream:
        _log_chain_event("retrieval_started", "streaming retrieval started")
        # 流式模式下，先跑到“检索 + 重排”阶段，把上下文先拿到手。
        # 这样前端可以尽早看到引用和片段信息，然后再渐进接收答案 token。
        state = await run_retrieval_only(
            runtime,
            question=body.question,
            conversation_id=body.conversation_id,
            history_messages=[item.model_dump(mode="json") for item in body.history_messages],
            top_k_sparse=min(body.top_k * 2, runtime.settings.bm25_top_k + 10),
            top_k_dense=min(body.top_k * 2, runtime.settings.dense_top_k + 10),
            rerank_top_n=body.top_k,
            user_context=user_context,
            access_filters=access_filters,
            risk_state=risk_state,
            audit_id=audit_id,
            disable_fast_path=_should_disable_fast_path(runtime, body),
        )
        state = _normalize_state_defaults(state, audit_id=audit_id)
        state["trace_id"] = trace_id
        update_request_log_context(
            data_classification=state.get("data_classification"),
            model_route=state.get("model_route"),
        )
        _log_chain_event(
            "retrieval_completed",
            "streaming retrieval completed",
            retrieved_chunks=len(state.get("reranked_hits") or []),
            refusal=bool(state.get("refusal")),
        )

        async def gen():
            """按 NDJSON 协议流式产出 meta / token / final 三类事件。"""

            ctx_raw = state.get("reranked_hits") or []
            contexts = [RetrievedChunk.model_validate(x) for x in ctx_raw]
            rq = state.get("rewritten_query") or state.get("resolved_query") or body.question
            refusal = bool(state.get("refusal"))
            meta = {
                "citations": state.get("citations") or [],
                "retrieved_chunks": ctx_raw,
                "confidence": state.get("confidence"),
                "refusal": refusal,
                "fast_path_source": state.get("fast_path_source"),
                **_response_security_fields(state, audit_id=audit_id),
            }
            # `meta` 事件让前端在答案尚未生成完时，也能先展示片段和引用信息。
            yield json.dumps({"type": "meta", "data": meta}, ensure_ascii=False) + "\n"
            if state.get("fast_path_source") or not contexts or refusal:
                # 空上下文或已拒答时，不再调用 LLM，直接把现有答案作为 token / final 发出。
                log_audit_event(
                    stage="response_sent",
                    audit_id=audit_id,
                    question=body.question,
                    user_context=user_context,
                    settings=runtime.settings,
                    state=state,
                    output=state.get("answer") or "",
                    extra={"stream": True},
                )
                _log_chain_event(
                    "response_sent",
                    "streaming response sent without generation",
                    refusal=bool(state.get("refusal")),
                )
                _emit_security_alert(
                    runtime=runtime,
                    audit_id=audit_id,
                    question=body.question,
                    user_context=user_context,
                    state=state,
                    stream=True,
                )
                yield json.dumps(
                    {"type": "token", "data": state.get("answer") or ""}, ensure_ascii=False
                ) + "\n"
                yield json.dumps(
                    {
                        "type": "final",
                        "data": {
                            "answer": state.get("answer") or "",
                            "confidence": float(state.get("confidence") or 0.0),
                            "citations": state.get("citations") or [],
                            "fast_path_source": state.get("fast_path_source"),
                            **_response_security_fields(state, audit_id=audit_id),
                        },
                    },
                    ensure_ascii=False,
                ) + "\n"
                return
            max_score = max(c.score for c in contexts)
            if max_score < runtime.settings.min_rerank_score:
                # 即便召回到了内容，如果最高重排分仍然过低，也视为“不足以可靠作答”。
                msg = "检索到的内容与问题相关性不足，无法可靠作答。"
                log_audit_event(
                    stage="response_sent",
                    audit_id=audit_id,
                    question=body.question,
                    user_context=user_context,
                    settings=runtime.settings,
                    state=state,
                    output=msg,
                    extra={"stream": True, "refusal_reason": "low_rerank_score"},
                )
                _log_chain_event(
                    "response_sent",
                    "streaming refusal sent after rerank gate",
                    refusal=True,
                    refusal_reason="low_rerank_score",
                )
                _emit_security_alert(
                    runtime=runtime,
                    audit_id=audit_id,
                    question=body.question,
                    user_context=user_context,
                    state={
                        **state,
                        "refusal": True,
                        "refusal_reason": "low_rerank_score",
                    },
                    stream=True,
                )
                yield json.dumps({"type": "token", "data": msg}, ensure_ascii=False) + "\n"
                yield json.dumps(
                    {
                        "type": "final",
                        "data": {
                            "answer": msg,
                            "confidence": 0.15,
                            "citations": [],
                            **_response_security_fields(
                                state,
                                audit_id=audit_id,
                                answer_mode="refusal",
                                refusal=True,
                                refusal_reason="low_rerank_score",
                            ),
                        },
                    },
                    ensure_ascii=False,
                ) + "\n"
                return
            prepared_contexts, egress = prepare_contexts_for_generation(
                contexts,
                settings=runtime.settings,
                data_classification=state.get("data_classification"),
                model_route=state.get("model_route"),
            )
            # 这里先做出域治理，再决定是否调用外部模型，
            # 目的是把“最小必要出域”落在生成前，而不是生成后补救。
            if (
                str(state.get("model_route") or "").strip().lower() == "local_only"
                and getattr(runtime.settings, "enable_local_fallback_generation", True)
            ):
                raw = build_local_grounded_output(
                    question=rq,
                    contexts=contexts,
                    conflict_summary=state.get("conflict_summary"),
                )
                log_audit_event(
                    stage="local_generation",
                    audit_id=audit_id,
                    question=body.question,
                    user_context=user_context,
                    settings=runtime.settings,
                    state=state,
                    output=raw,
                    extra={"stream": True, "execution_mode": "local_fallback"},
                )
                _log_chain_event(
                    "generation_completed",
                    "local fallback generation completed",
                    execution_mode="local_fallback",
                )
                answer, conf, reasoning, citations = parse_llm_grounded_output(raw, contexts)
                final_state = {
                    **state,
                    "answer_mode": "local_grounded_answer",
                    "refusal": False,
                    "refusal_reason": "",
                }
                log_audit_event(
                    stage="response_sent",
                    audit_id=audit_id,
                    question=body.question,
                    user_context=user_context,
                    settings=runtime.settings,
                    state=final_state,
                    output=answer,
                    extra={"stream": True, "execution_mode": "local_fallback"},
                )
                _log_chain_event(
                    "response_sent",
                    "streaming response sent",
                    execution_mode="local_fallback",
                )
                _emit_security_alert(
                    runtime=runtime,
                    audit_id=audit_id,
                    question=body.question,
                    user_context=user_context,
                    state=final_state,
                    stream=True,
                )
                yield json.dumps({"type": "token", "data": answer}, ensure_ascii=False) + "\n"
                yield json.dumps(
                    {
                        "type": "final",
                        "data": {
                            "answer": answer,
                            "confidence": conf,
                            "reasoning_summary": reasoning,
                            "citations": [c.model_dump() for c in citations],
                            **_response_security_fields(
                                final_state,
                                audit_id=audit_id,
                                answer_mode="local_grounded_answer",
                            ),
                        },
                    },
                    ensure_ascii=False,
                ) + "\n"
                return
            if not egress.get("allowed"):
                # 即使已经检索到证据，只要出域策略不允许，也必须拒答或走本地降级，
                # 不能为了“回答完整”强行把敏感上下文送给外部模型。
                msg = "当前问题涉及高敏数据，禁止将原始内容发送到外部模型。"
                log_audit_event(
                    stage="response_sent",
                    audit_id=audit_id,
                    question=body.question,
                    user_context=user_context,
                    settings=runtime.settings,
                    state=state,
                    output=msg,
                    extra={"stream": True, "refusal_reason": egress.get("refusal_reason")},
                )
                _log_chain_event(
                    "response_sent",
                    "streaming refusal sent by egress policy",
                    refusal=True,
                    refusal_reason=str(egress.get("refusal_reason") or "restricted_data_local_only"),
                )
                _emit_security_alert(
                    runtime=runtime,
                    audit_id=audit_id,
                    question=body.question,
                    user_context=user_context,
                    state={
                        **state,
                        "refusal": True,
                        "refusal_reason": str(
                            egress.get("refusal_reason") or "restricted_data_local_only"
                        ),
                    },
                    stream=True,
                )
                yield json.dumps({"type": "token", "data": msg}, ensure_ascii=False) + "\n"
                yield json.dumps(
                    {
                        "type": "final",
                        "data": {
                            "answer": msg,
                            "confidence": 0.0,
                            "citations": [],
                            **_response_security_fields(
                                state,
                                audit_id=audit_id,
                                answer_mode="refusal",
                                refusal=True,
                                refusal_reason=str(
                                    egress.get("refusal_reason") or "restricted_data_local_only"
                                ),
                            ),
                        },
                    },
                    ensure_ascii=False,
                ) + "\n"
                return
            # 把检索片段格式化为带 chunk_id 的上下文块，便于 LLM 在答案中显式引用。
            ctx_text = format_context_blocks(prepared_contexts)
            user = f"QUESTION:\n{rq}\n\nCONTEXT:\n{ctx_text}"
            log_audit_event(
                stage="prompt_audited",
                audit_id=audit_id,
                question=body.question,
                user_context=user_context,
                settings=runtime.settings,
                state=state,
                prompt=user,
                extra={"stream": True, "egress_strategy": egress.get("strategy")},
            )
            _log_chain_event(
                "generation_started",
                "external answer generation started",
                egress_strategy=egress.get("strategy"),
                context_count=len(prepared_contexts),
            )
            messages = [
                {"role": "system", "content": GROUNDED_ANSWER_SYSTEM},
                {"role": "user", "content": user},
            ]
            # 流式场景直接单独 new 一个 LLMClient，避免把 Graph 的非流式 complete 接口耦合进来。
            llm = LLMClient(runtime.settings)
            buf: list[str] = []
            async for tok in llm.stream(
                messages,
                task="answer_generation",
                temperature=0.1,
                max_tokens=1024,
            ):
                buf.append(tok)
                yield json.dumps({"type": "token", "data": tok}, ensure_ascii=False) + "\n"
            raw = "".join(buf)
            log_audit_event(
                stage="output_audited",
                audit_id=audit_id,
                question=body.question,
                user_context=user_context,
                settings=runtime.settings,
                state=state,
                output=raw,
                extra={"stream": True, "egress_strategy": egress.get("strategy")},
            )
            _log_chain_event(
                "generation_completed",
                "external answer generation completed",
                egress_strategy=egress.get("strategy"),
            )
            # 收到完整原始输出后，再统一解析 answer / confidence / citations。
            answer, conf, reasoning, citations = parse_llm_grounded_output(raw, prepared_contexts)
            log_audit_event(
                stage="response_sent",
                audit_id=audit_id,
                question=body.question,
                user_context=user_context,
                settings=runtime.settings,
                state=state,
                output=answer,
                extra={"stream": True, "egress_strategy": egress.get("strategy")},
            )
            _log_chain_event("response_sent", "streaming response sent")
            _emit_security_alert(
                runtime=runtime,
                audit_id=audit_id,
                question=body.question,
                user_context=user_context,
                state=state,
                stream=True,
            )
            yield json.dumps(
                {
                    "type": "final",
                    "data": {
                        "answer": answer,
                        "confidence": conf,
                        "reasoning_summary": reasoning,
                        "citations": [c.model_dump() for c in citations],
                        **_response_security_fields(
                            state,
                            audit_id=audit_id,
                            answer_mode="grounded_answer",
                        ),
                    },
                },
                ensure_ascii=False,
            ) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    # 非流式模式更简单：直接等待完整图执行完成，再一次性返回结构化响应。
    _log_chain_event("graph_started", "non-stream graph execution started")
    state = await _run_graph(
        runtime,
        body,
        user_context=user_context,
        access_filters=access_filters,
        risk_state=risk_state,
        audit_id=audit_id,
    )
    state = _normalize_state_defaults(state, audit_id=audit_id)
    state["trace_id"] = trace_id
    update_request_log_context(
        data_classification=state.get("data_classification"),
        model_route=state.get("model_route"),
    )
    _log_chain_event(
        "graph_completed",
        "non-stream graph execution completed",
        retrieved_chunks=len(state.get("reranked_hits") or []),
        refusal=bool(state.get("refusal")),
        fast_path_source=state.get("fast_path_source"),
    )
    resp = ChatResponse(
        answer=state.get("answer") or "",
        confidence=float(state.get("confidence") or 0.0),
        fast_path_source=state.get("fast_path_source") or None,
        citations=_citations_from_state(state),
        retrieved_chunks=_chunks_from_state(state),
        **_response_security_fields(state, audit_id=audit_id),
    )
    log_audit_event(
        stage="response_sent",
        audit_id=audit_id,
        question=body.question,
        user_context=user_context,
        settings=runtime.settings,
        state=state,
        output=resp.answer,
        extra={"stream": False},
    )
    _log_chain_event("response_sent", "non-stream response sent")
    _emit_security_alert(
        runtime=runtime,
        audit_id=audit_id,
        question=body.question,
        user_context=user_context,
        state=state,
        stream=False,
    )
    return JSONResponse(resp.model_dump())
