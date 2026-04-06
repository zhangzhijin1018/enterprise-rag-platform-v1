"""apps.worker.jobs 包初始化模块。用于标记包边界，并为同目录下的实现提供统一导入入口。"""

from apps.worker.jobs.ingest_job import run_ingest_path

__all__ = ["run_ingest_path"]
