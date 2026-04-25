#!/usr/bin/env python3
"""
将 data/mock_corpus 下的 Markdown 解析、向量化并写入 Milvus。

用法（在仓库根目录）:
  conda activate tmf_project
  python infra/scripts/seed_mock_index.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 仓库根：enterprise-rag-platform/
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("VECTOR_BACKEND", "milvus")
os.environ.setdefault("MILVUS_URI", str(ROOT / "data" / "milvus" / "enterprise_rag.db"))
os.makedirs(Path(os.environ["MILVUS_URI"]).parent, exist_ok=True)

from core.ingestion.pipeline import index_chunks, parse_and_chunk_file  # noqa: E402
from core.services.runtime import get_runtime, reset_runtime  # noqa: E402


def main() -> None:
    corpus = ROOT / "data" / "mock_corpus"
    if not corpus.is_dir():
        raise SystemExit(f"Missing corpus dir: {corpus}")

    reset_runtime()
    runtime = get_runtime()

    files = sorted(corpus.glob("*.md"))
    if not files:
        raise SystemExit(f"No .md files in {corpus}")

    first = True
    for path in files:
        _, chunks = parse_and_chunk_file(path, source=f"mock://{path.name}")
        # 第一份文档使用 replace_all，后续文档走增量合并，便于快速重建一份演示索引。
        index_chunks(runtime, chunks, replace_all=first)
        first = False
        print(f"Indexed {path.name}: {len(chunks)} chunks")

    total = len(runtime.dense.fetch_all_chunks())
    print(f"Done. Chunks total: {total}")
    print(f"Milvus URI: {os.environ['MILVUS_URI']}")


if __name__ == "__main__":
    main()
