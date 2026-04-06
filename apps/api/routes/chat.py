"""问答接口路由模块。

这个文件是前端调用最频繁的入口：
- 非流式模式走完整 LangGraph 问答链路后一次性返回；
- 流式模式先跑检索，再把 LLM token 通过 NDJSON 持续推给前端。
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from apps.api.dependencies.common import get_runtime_dep
from apps.api.schemas.chat import ChatRequest, ChatResponse, CitationSchema, RetrievedChunkSchema
from core.generation.answer_builder import parse_llm_grounded_output
from core.generation.context_format import format_context_blocks
from core.generation.llm_client import LLMClient
from core.generation.prompts.templates import GROUNDED_ANSWER_SYSTEM
from core.orchestration.graph import run_rag_async
from core.orchestration.retrieval_pipeline import run_retrieval_only
from core.retrieval.schemas import RetrievedChunk
from core.services.runtime import RAGRuntime

router = APIRouter(prefix="/chat", tags=["chat"])


def _chunks_from_state(state: dict) -> list[RetrievedChunkSchema]:
    """把图状态里的 chunk 字典列表转成 API schema。"""

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
    """把图状态里的引用列表转成对外响应模型。"""

    return [CitationSchema.model_validate(c) for c in (state.get("citations") or [])]


async def _run_graph(runtime: RAGRuntime, body: ChatRequest) -> dict:
    """执行完整 RAG 图，并根据前端 `top_k` 派生召回参数。

    设计原因：
    - 前端只感知一个 `top_k`，避免暴露太多底层参数。
    - 内部会把召回数量放大，再交给 rerank 收敛到最终的 top_n，
      这样能兼顾召回率和最终答案质量。
    """

    sk = min(body.top_k * 2, runtime.settings.bm25_top_k + 10)
    dk = min(body.top_k * 2, runtime.settings.dense_top_k + 10)
    return await run_rag_async(
        runtime,
        question=body.question,
        conversation_id=body.conversation_id,
        top_k_sparse=sk,
        top_k_dense=dk,
        rerank_top_n=body.top_k,
    )


@router.post("")
async def chat(
    body: ChatRequest,
    runtime: RAGRuntime = Depends(get_runtime_dep),
):
    """统一处理流式与非流式问答请求。"""

    if body.stream:
        # 流式模式下，先跑到“检索 + 重排”阶段，把上下文先拿到手。
        # 这样前端可以尽早看到引用和片段信息，然后再渐进接收答案 token。
        state = await run_retrieval_only(
            runtime,
            question=body.question,
            conversation_id=body.conversation_id,
            top_k_sparse=min(body.top_k * 2, runtime.settings.bm25_top_k + 10),
            top_k_dense=min(body.top_k * 2, runtime.settings.dense_top_k + 10),
            rerank_top_n=body.top_k,
        )

        async def gen():
            """按 NDJSON 协议流式产出 meta / token / final 三类事件。"""

            ctx_raw = state.get("reranked_hits") or []
            contexts = [RetrievedChunk.model_validate(x) for x in ctx_raw]
            rq = state.get("rewritten_query") or body.question
            refusal = bool(state.get("refusal"))
            meta = {
                "citations": state.get("citations") or [],
                "retrieved_chunks": ctx_raw,
                "confidence": state.get("confidence"),
                "refusal": refusal,
            }
            # `meta` 事件让前端在答案尚未生成完时，也能先展示片段和引用信息。
            yield json.dumps({"type": "meta", "data": meta}, ensure_ascii=False) + "\n"
            if not contexts or refusal:
                # 空上下文或已拒答时，不再调用 LLM，直接把现有答案作为 token / final 发出。
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
                        },
                    },
                    ensure_ascii=False,
                ) + "\n"
                return
            max_score = max(c.score for c in contexts)
            if max_score < runtime.settings.min_rerank_score:
                # 即便召回到了内容，如果最高重排分仍然过低，也视为“不足以可靠作答”。
                msg = "检索到的内容与问题相关性不足，无法可靠作答。"
                yield json.dumps({"type": "token", "data": msg}, ensure_ascii=False) + "\n"
                yield json.dumps(
                    {
                        "type": "final",
                        "data": {
                            "answer": msg,
                            "confidence": 0.15,
                            "citations": [],
                        },
                    },
                    ensure_ascii=False,
                ) + "\n"
                return
            # 把检索片段格式化为带 chunk_id 的上下文块，便于 LLM 在答案中显式引用。
            ctx_text = format_context_blocks(contexts)
            user = f"QUESTION:\n{rq}\n\nCONTEXT:\n{ctx_text}"
            messages = [
                {"role": "system", "content": GROUNDED_ANSWER_SYSTEM},
                {"role": "user", "content": user},
            ]
            # 流式场景直接单独 new 一个 LLMClient，避免把 Graph 的非流式 complete 接口耦合进来。
            llm = LLMClient(runtime.settings)
            buf: list[str] = []
            async for tok in llm.stream(messages, temperature=0.1, max_tokens=1024):
                buf.append(tok)
                yield json.dumps({"type": "token", "data": tok}, ensure_ascii=False) + "\n"
            raw = "".join(buf)
            # 收到完整原始输出后，再统一解析 answer / confidence / citations。
            answer, conf, reasoning, citations = parse_llm_grounded_output(raw, contexts)
            yield json.dumps(
                {
                    "type": "final",
                    "data": {
                        "answer": answer,
                        "confidence": conf,
                        "reasoning_summary": reasoning,
                        "citations": [c.model_dump() for c in citations],
                    },
                },
                ensure_ascii=False,
            ) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    # 非流式模式更简单：直接等待完整图执行完成，再一次性返回结构化响应。
    state = await _run_graph(runtime, body)
    resp = ChatResponse(
        answer=state.get("answer") or "",
        confidence=float(state.get("confidence") or 0.0),
        citations=_citations_from_state(state),
        retrieved_chunks=_chunks_from_state(state),
    )
    return JSONResponse(resp.model_dump())
