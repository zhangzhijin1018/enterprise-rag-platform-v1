"""评测接口路由模块。负责触发 RAGAS 评测，并把报告路径和摘要返回给调用方。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends

from apps.api.dependencies.common import get_runtime_dep
from apps.api.schemas.eval_schema import EvalRequest, EvalResponse
from core.config.settings import get_settings
from core.evaluation.ragas_runner import run_ragas_eval_async
from core.services.runtime import RAGRuntime

router = APIRouter(tags=["evaluation"])


@router.post("/eval", response_model=EvalResponse)
async def run_eval(
    body: EvalRequest | None = None,
    runtime: RAGRuntime = Depends(get_runtime_dep),
) -> EvalResponse:
    body = body or EvalRequest()
    settings = get_settings()
    ds = body.dataset_path or Path(settings.eval_dataset_path)
    out = await run_ragas_eval_async(dataset_path=ds, output_dir=Path(settings.eval_output_dir), runtime=runtime)
    summary = None
    if out.is_file():
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
            summary = data.get("summary")
        except json.JSONDecodeError:
            summary = None
    return EvalResponse(report_path=str(out), summary=summary)
