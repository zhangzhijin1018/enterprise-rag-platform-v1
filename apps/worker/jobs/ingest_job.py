"""Worker 入库任务模块。负责把磁盘文件接入解析、切块、向量化和索引持久化流程。"""

from __future__ import annotations

from pathlib import Path

from core.ingestion.pipeline import index_chunks, parse_and_chunk_file
from core.services.runtime import RAGRuntime


def run_ingest_path(
    runtime: RAGRuntime,
    path: Path,
    *,
    source: str | None = None,
    replace_all: bool = False,
) -> None:
    """执行单文件入库。

    这是 worker 侧最短主路径：
    - parse_and_chunk_file 负责把原始文件变成 chunks
    - index_chunks 负责把 chunks 写入索引并刷新运行时
    """

    _, chunks = parse_and_chunk_file(path, source=source)
    index_chunks(runtime, chunks, replace_all=replace_all)
