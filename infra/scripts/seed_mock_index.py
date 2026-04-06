#!/usr/bin/env python3
"""
将 data/mock_corpus 下的 Markdown 解析、向量化并写入 VECTOR_STORE_PATH。
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

os.environ.setdefault("VECTOR_STORE_PATH", str(ROOT / "data" / "vector_store"))
os.makedirs(os.environ["VECTOR_STORE_PATH"], exist_ok=True)

from core.ingestion.pipeline import index_chunks, parse_and_chunk_file  # noqa: E402
from core.services.runtime import get_runtime, reset_runtime  # noqa: E402


def main() -> None:
    corpus = ROOT / "data" / "mock_corpus"
    if not corpus.is_dir():
        raise SystemExit(f"Missing corpus dir: {corpus}")

    reset_runtime()
    runtime = get_runtime()
    runtime.store.clear()

    files = sorted(corpus.glob("*.md"))
    if not files:
        raise SystemExit(f"No .md files in {corpus}")

    for path in files:
        _, chunks = parse_and_chunk_file(path, source=f"mock://{path.name}")
        index_chunks(runtime, chunks, replace_all=False)
        print(f"Indexed {path.name}: {len(chunks)} chunks")

    print(f"Done. Chunks total: {runtime.store.chunk_count}")
    print(f"Store: {os.environ['VECTOR_STORE_PATH']}")


if __name__ == "__main__":
    main()
