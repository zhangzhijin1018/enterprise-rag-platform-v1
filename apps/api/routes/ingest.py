"""入库与索引管理路由模块。

这个文件把“上传文件”“查询任务状态”“重建索引”三个常见运维动作对外暴露成 HTTP 接口。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile

from apps.api.dependencies.common import get_runtime_dep
from apps.api.job_store import job_store
from apps.api.schemas.common import IngestResponse, JobStatusResponse
from apps.worker.jobs.ingest_job import run_ingest_path
from core.ingestion.pipeline import rebuild_index_from_store_files
from core.services.runtime import RAGRuntime

router = APIRouter(tags=["ingest"])


def _process_upload(job_id: str, path: Path, runtime: RAGRuntime) -> None:
    """后台处理上传文件。

    之所以放到后台任务里，是因为解析、切块、向量化可能比较慢，
    不适合阻塞 HTTP 请求连接一直等待。
    """

    try:
        job_store.update(job_id, "running")
        run_ingest_path(runtime, path, source=str(path.name))
        job_store.update(job_id, "completed")
    except Exception as e:  # noqa: BLE001
        # 这里把异常信息写回 job_store，前端轮询任务状态时就能看到失败原因。
        job_store.update(job_id, "failed", detail=str(e))
    finally:
        try:
            # 临时上传文件只在入库期间有价值，处理完成后立刻清理。
            path.unlink(missing_ok=True)
        except OSError:
            pass


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    runtime: RAGRuntime = Depends(get_runtime_dep),
) -> IngestResponse:
    """上传单个文件并异步触发入库。"""

    job_id = job_store.create()
    suffix = Path(file.filename or "upload").suffix
    # 先把 FastAPI 上传流落成临时文件，再交给后台任务处理，避免把文件内容长期放在内存中。
    tmp = Path(tempfile.mkdtemp()) / f"upload{suffix}"
    content = await file.read()
    tmp.write_bytes(content)
    background.add_task(_process_upload, job_id, tmp, runtime)
    return IngestResponse(job_id=job_id, status="accepted")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    """查询后台任务状态。"""

    st = job_store.get(job_id)
    if not st:
        return JobStatusResponse(job_id=job_id, status="unknown")
    return JobStatusResponse(job_id=job_id, status=st["status"], detail=st.get("detail"))


@router.post("/reindex", response_model=IngestResponse)
def reindex(
    background: BackgroundTasks,
    runtime: RAGRuntime = Depends(get_runtime_dep),
) -> IngestResponse:
    """根据磁盘里的 chunks 重新生成向量索引。"""

    job_id = job_store.create()

    def work() -> None:
        # 这里和上传入库一样走后台任务，避免重建向量时阻塞请求线程。
        try:
            job_store.update(job_id, "running")
            rebuild_index_from_store_files(runtime)
            job_store.update(job_id, "completed")
        except Exception as e:  # noqa: BLE001
            job_store.update(job_id, "failed", detail=str(e))

    background.add_task(work)
    return IngestResponse(job_id=job_id, status="accepted")
