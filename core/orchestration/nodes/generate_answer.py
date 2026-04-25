"""答案生成节点模块。

它负责拿到最终重排结果后：
1. 做最后一次“是否足够可靠”的门控；
2. 组织 grounded prompt；
3. 调用 LLM；
4. 解析结构化答案与引用。
"""

from __future__ import annotations

from core.generation.answer_builder import parse_llm_grounded_output
from core.generation.context_format import format_context_blocks, select_contexts_for_prompt
from core.generation.egress_policy import prepare_contexts_for_generation
from core.generation.local_executor import build_local_grounded_output
from core.generation.prompts.templates import GROUNDED_ANSWER_SYSTEM
from core.observability import get_logger
from core.observability.audit import log_audit_event
from core.orchestration.state import RAGState
from core.retrieval.schemas import RetrievedChunk
from core.security.risk_engine import (
    RuleBasedRiskEngine,
    build_risk_context,
    decision_to_state_update,
    safe_evaluate_risk,
)
from core.services.runtime import RAGRuntime

logger = get_logger(__name__)


async def generate_answer_node(state: RAGState, runtime: RAGRuntime) -> RAGState:
    """执行回答生成节点。

    这一层是“检索”到“回答”的最后一道收口：
    1. 检查上下文是否足够可靠
    2. 执行生成前风控
    3. 应用出域策略和上下文压缩
    4. 调用最终回答模型或本地受限模式
    5. 解析答案、置信度和引用
    """

    logger.info("generation started", extra={"event": "generation_started"})
    rq = (
        state.get("rewritten_query")
        or state.get("resolved_query")
        or state.get("question")
        or ""
    )
    ctx_raw = state.get("reranked_hits") or []
    contexts = [RetrievedChunk.model_validate(x) for x in ctx_raw]
    if not contexts:
        # 没有上下文时直接拒答，避免模型凭空编造答案。
        result = {
            "answer": "上下文不足，已拒答。",
            "confidence": 0.0,
            "reasoning_summary": "无可用上下文。",
            "citations": [],
            "refusal": True,
            "refusal_reason": "empty_context",
            "grounding_ok": False,
        }
        logger.info("generation completed", extra={"event": "generation_completed", "refusal": True})
        return result

    max_score = max(c.score for c in contexts)
    if max_score < runtime.settings.min_rerank_score:
        # 即使有上下文，也要保证它和问题足够相关。
        result = {
            "answer": "检索到的内容与问题相关性不足，无法可靠作答。",
            "confidence": 0.15,
            "reasoning_summary": "重排分数低于阈值。",
            "citations": [],
            "refusal": True,
            "refusal_reason": "low_relevance",
            "grounding_ok": False,
        }
        logger.info("generation completed", extra={"event": "generation_completed", "refusal": True})
        return result

    risk_context = build_risk_context(
        stage="generation",
        question=rq,
        audit_id=str(state.get("audit_id") or ""),
        user_context=state.get("user_context") or {},
        state=state,
    )
    risk_engine = getattr(runtime, "risk_engine", RuleBasedRiskEngine(runtime.settings))
    risk_decision = safe_evaluate_risk(risk_engine, risk_context, runtime.settings)
    risk_state = decision_to_state_update(risk_decision)
    # `effective_model_route` 表示这一轮生成阶段最终采用的执行路线。
    # 如果风控要求强制 local_only，会覆盖上游 retrieval 给出的 model_route。
    effective_model_route = (
        "local_only" if risk_decision.force_local_only else str(state.get("model_route") or "")
    )
    if not risk_decision.allow:
        result = {
            "answer": "当前请求命中了企业风控策略，已拒答。",
            "confidence": 0.0,
            "reasoning_summary": "生成前风控引擎阻止了后续处理。",
            "citations": [],
            "refusal": True,
            "refusal_reason": risk_decision.reason or "risk_engine_denied",
            "answer_mode": "refusal",
            "grounding_ok": False,
            **risk_state,
        }
        logger.info("generation completed", extra={"event": "generation_completed", "refusal": True})
        return result

    if (
        effective_model_route.strip().lower() == "local_only"
        and getattr(runtime.settings, "enable_local_fallback_generation", True)
    ):
        # 对 restricted / local_only 场景，优先走本地受限模式占位执行，
        # 避免把高敏上下文直接送到外部模型。
        raw = build_local_grounded_output(
            question=rq,
            contexts=contexts,
            conflict_summary=state.get("conflict_summary"),
        )
        log_audit_event(
            stage="local_generation",
            audit_id=str(state.get("audit_id") or ""),
            question=rq,
            user_context=state.get("user_context") or {},
            settings=runtime.settings,
            state=state,
            output=raw,
            extra={"execution_mode": "local_fallback"},
        )
        answer, conf, reasoning, citations = parse_llm_grounded_output(raw, contexts)
        result = {
            "answer": answer,
            "confidence": conf,
            "reasoning_summary": reasoning,
            "citations": [c.model_dump(mode="json") for c in citations],
            "refusal": False,
            "refusal_reason": "",
            "answer_mode": "local_grounded_answer",
            "model_route": effective_model_route,
            **risk_state,
        }
        logger.info(
            "generation completed",
            extra={"event": "generation_completed", "model_route": effective_model_route, "refusal": False},
        )
        return result

    prepared_contexts, egress = prepare_contexts_for_generation(
        contexts,
        settings=runtime.settings,
        data_classification=state.get("data_classification"),
        model_route=effective_model_route,
    )
    # egress policy 负责把“数据分级”和“模型路由”转成真正的出域动作：
    # - 允许完整上下文
    # - 脱敏后出域
    # - 只传最小必要片段
    # - 直接拒答
    if not egress.get("allowed"):
        result = {
            "answer": "当前问题涉及高敏数据，禁止将原始内容发送到外部模型。",
            "confidence": 0.0,
            "reasoning_summary": "数据分级策略阻止出域生成。",
            "citations": [],
            "refusal": True,
            "refusal_reason": str(egress.get("refusal_reason") or "restricted_data_local_only"),
            "answer_mode": "refusal",
            "grounding_ok": False,
            "model_route": effective_model_route,
            **risk_state,
        }
        logger.info("generation completed", extra={"event": "generation_completed", "refusal": True})
        return result
    if not prepared_contexts:
        result = {
            "answer": "可用于生成的上下文为空，已拒答。",
            "confidence": 0.0,
            "reasoning_summary": "上下文在出域策略阶段被清空。",
            "citations": [],
            "refusal": True,
            "refusal_reason": "empty_egress_context",
            "answer_mode": "refusal",
            "grounding_ok": False,
            "model_route": effective_model_route,
            **risk_state,
        }
        logger.info("generation completed", extra={"event": "generation_completed", "refusal": True})
        return result

    packed_contexts = select_contexts_for_prompt(
        prepared_contexts,
        max_docs=int(getattr(runtime.settings, "generation_context_max_docs", 4)),
        max_chunks_per_doc=int(
            getattr(runtime.settings, "generation_context_max_chunks_per_doc", 2)
        ),
        max_chars=int(getattr(runtime.settings, "generation_context_max_chars", 3200)),
    )
    if not packed_contexts:
        result = {
            "answer": "上下文压缩后为空，已拒答。",
            "confidence": 0.0,
            "reasoning_summary": "context packing 后没有保留可用证据。",
            "citations": [],
            "refusal": True,
            "refusal_reason": "empty_packed_context",
            "answer_mode": "refusal",
            "grounding_ok": False,
            "model_route": effective_model_route,
            **risk_state,
        }
        logger.info("generation completed", extra={"event": "generation_completed", "refusal": True})
        return result

    # 把经过风控和压缩后的证据拼成 prompt 上下文块，
    # 后续模型只允许基于这些块回答。
    ctx_text = format_context_blocks(packed_contexts)
    governance_notice = ""
    if state.get("conflict_detected") and state.get("conflict_summary"):
        # 当治理层已经检测到冲突时，把冲突摘要显式喂给模型，
        # 避免模型把多版本证据强行糊成一个“看起来很顺”的答案。
        governance_notice = f"GOVERNANCE_NOTICE:\n{state.get('conflict_summary')}\n\n"
    user = f"QUESTION:\n{rq}\n\n{governance_notice}CONTEXT:\n{ctx_text}"
    log_audit_event(
        stage="prompt_audited",
        audit_id=str(state.get("audit_id") or ""),
        question=rq,
        user_context=state.get("user_context") or {},
        settings=runtime.settings,
        state=state,
        prompt=user,
        extra={
            "egress_strategy": egress.get("strategy"),
            "packed_context_count": len(packed_contexts),
            "original_context_count": len(prepared_contexts),
        },
    )
    messages = [
        {"role": "system", "content": GROUNDED_ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]
    raw, _ = await runtime.llm.complete(
        messages,
        task="answer_generation",
        temperature=0.1,
        max_tokens=1024,
    )
    log_audit_event(
        stage="output_audited",
        audit_id=str(state.get("audit_id") or ""),
        question=rq,
        user_context=state.get("user_context") or {},
        settings=runtime.settings,
        state=state,
        output=raw,
        extra={
            "egress_strategy": egress.get("strategy"),
            "packed_context_count": len(packed_contexts),
            "original_context_count": len(prepared_contexts),
        },
    )
    # 统一解析 LLM 输出，抽取 answer / confidence / citations。
    # 这样无论最终回答来自外部模型还是后续替换的本地模型，
    # 对上层 API 暴露的结构都保持一致。
    answer, conf, reasoning, citations = parse_llm_grounded_output(raw, packed_contexts)

    result = {
        "answer": answer,
        "confidence": conf,
        "reasoning_summary": reasoning,
        "citations": [c.model_dump(mode="json") for c in citations],
        "refusal": False,
        "refusal_reason": "",
        "answer_mode": str(state.get("answer_mode") or "grounded_answer"),
        "model_route": effective_model_route,
        **risk_state,
    }
    logger.info(
        "generation completed",
        extra={"event": "generation_completed", "model_route": effective_model_route, "refusal": False},
    )
    return result
