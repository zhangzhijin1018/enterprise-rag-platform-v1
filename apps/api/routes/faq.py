"""FAQ 管理接口。

这层不直接参与主 RAG 图，但它决定了 fast path 里的 FAQ 数据怎么被导入、查看、启停和更新。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from apps.api.dependencies.common import get_runtime_dep
from apps.api.schemas.faq import (
    FaqImportResponse,
    FaqItemSchema,
    FaqListResponse,
    FaqToggleRequest,
    FaqToggleResponse,
    FaqUpdateRequest,
    FaqUpdateResponse,
)
from core.services.runtime import RAGRuntime

router = APIRouter(prefix="/faq", tags=["faq"])


@router.post("/import", response_model=FaqImportResponse)
async def import_faq_csv(
    file: UploadFile = File(...),
    runtime: RAGRuntime = Depends(get_runtime_dep),
) -> FaqImportResponse:
    """导入 FAQ CSV，并立即刷新 FAQ 检索索引。"""

    suffix = Path(file.filename or "faq.csv").suffix or ".csv"
    tmp = Path(tempfile.mkdtemp()) / f"faq_import{suffix}"
    try:
        # FAQ 导入完成后必须立刻刷新 FAQ 检索器，否则 fast path 还是旧数据。
        tmp.write_bytes(await file.read())
        imported = runtime.faq_store.import_csv(tmp)
        runtime.reload_faq_index()
        return FaqImportResponse(imported=imported, status="completed")
    finally:
        tmp.unlink(missing_ok=True)


@router.get("", response_model=FaqListResponse)
def list_faq_entries(runtime: RAGRuntime = Depends(get_runtime_dep)) -> FaqListResponse:
    """列出当前 FAQ 全量条目，供前端管理页使用。"""

    items = [
        FaqItemSchema(
            id=entry.entry_id,
            question=entry.question,
            answer=entry.answer,
            keywords=entry.keywords,
            category=entry.category,
            enabled=entry.enabled,
            hit_count=entry.hit_count,
            last_hit_at=entry.last_hit_at,
        )
        for entry in runtime.faq_store.list_all_entries()
    ]
    return FaqListResponse(items=items)


@router.patch("/{entry_id}", response_model=FaqToggleResponse)
def toggle_faq_entry(
    entry_id: int,
    body: FaqToggleRequest,
    runtime: RAGRuntime = Depends(get_runtime_dep),
) -> FaqToggleResponse:
    """启用或停用 FAQ，并刷新运行时 FAQ 检索索引。"""

    ok = runtime.faq_store.set_enabled(entry_id, body.enabled)
    if not ok:
        raise HTTPException(status_code=404, detail="FAQ entry not found")
    # 启停会直接影响 fast path 行为，所以这里同步刷新 FAQ 内存索引。
    runtime.reload_faq_index()
    return FaqToggleResponse(id=entry_id, enabled=body.enabled, status="completed")


@router.put("/{entry_id}", response_model=FaqUpdateResponse)
def update_faq_entry(
    entry_id: int,
    body: FaqUpdateRequest,
    runtime: RAGRuntime = Depends(get_runtime_dep),
) -> FaqUpdateResponse:
    """更新 FAQ 的问题、答案、关键词和分类。"""

    ok = runtime.faq_store.update_entry(
        entry_id,
        question=body.question.strip(),
        answer=body.answer.strip(),
        keywords=body.keywords.strip(),
        category=body.category.strip(),
    )
    if not ok:
        raise HTTPException(status_code=404, detail="FAQ entry not found")
    # 更新 FAQ 内容后，同样要刷新 FAQ 内存索引。
    runtime.reload_faq_index()
    return FaqUpdateResponse(id=entry_id, status="completed")
