"""内存版任务状态存储模块。用于追踪入库与重建索引这类后台任务的执行状态。"""

from __future__ import annotations

import threading
import uuid
from typing import Any


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(self) -> str:
        jid = str(uuid.uuid4())
        with self._lock:
            self._jobs[jid] = {"status": "queued", "detail": None}
        return jid

    def update(self, job_id: str, status: str, detail: str | None = None) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = status
                self._jobs[job_id]["detail"] = detail

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._jobs.get(job_id, {})) if job_id in self._jobs else None


job_store = JobStore()
