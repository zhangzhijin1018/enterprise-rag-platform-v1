"""索引持久化模块。

这个模块把可检索数据拆成两类文件持久化：
- `chunks.jsonl`：保存 chunk 文本和 metadata；
- `embeddings.npy`：保存与 chunk 一一对应的向量矩阵。

这种设计简单、透明，特别适合教学、调试和本地离线联调。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from core.config.settings import Settings, get_settings
from core.models.document import TextChunk
from core.observability import get_logger

logger = get_logger(__name__)


class IndexStore:
    """Persists chunks and embedding matrix; BM25 rebuilt in SparseRetriever."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        # 向量索引目录由配置控制，默认落到 `./data/vector_store`。
        self._root = Path(self._settings.vector_store_path)
        self._chunks_path = self._root / "chunks.jsonl"
        self._emb_path = self._root / "embeddings.npy"
        self._meta_path = self._root / "index_meta.json"
        # 这里用可重入锁，方便未来某些方法在持锁状态下互相调用。
        self._lock = threading.RLock()
        self._chunks: list[TextChunk] = []
        self._embeddings: np.ndarray | None = None

    def _ensure_dir(self) -> None:
        """确保索引目录存在。"""

        self._root.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        """从磁盘加载 chunk 和向量矩阵。"""

        with self._lock:
            # 每次 load 都先清空内存，避免磁盘状态和旧缓存混在一起。
            self._chunks = []
            self._embeddings = None
            if self._chunks_path.is_file():
                for line in self._chunks_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    self._chunks.append(TextChunk.model_validate(obj))
            if self._emb_path.is_file() and self._chunks:
                self._embeddings = np.load(self._emb_path)
                # 最重要的一致性检查：向量行数必须和 chunk 数一致。
                if self._embeddings.shape[0] != len(self._chunks):
                    logger.warning(
                        "embedding rows mismatch chunks; clearing embeddings",
                        extra={"chunks": len(self._chunks), "emb": self._embeddings.shape[0]},
                    )
                    # 不一致时宁可清空向量，也不能把错位向量继续用于检索。
                    self._embeddings = None

    def save(self) -> None:
        """把当前内存索引写回磁盘。"""

        with self._lock:
            self._ensure_dir()
            # JSONL 的好处是简单直观，调试时可以直接 `cat` 看每个 chunk。
            lines = [c.model_dump_json() for c in self._chunks]
            self._chunks_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if self._embeddings is not None:
                # `npy` 适合存储矩阵，读写简单且速度快。
                np.save(self._emb_path, self._embeddings)
            self._meta_path.write_text(
                json.dumps({"num_chunks": len(self._chunks)}), encoding="utf-8"
            )

    def clear(self) -> None:
        """清空内存和磁盘上的索引文件。"""

        with self._lock:
            self._chunks = []
            self._embeddings = None
            for p in (self._chunks_path, self._emb_path, self._meta_path):
                if p.is_file():
                    p.unlink()

    def upsert_chunks(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        """按 chunk_id 增量更新 chunk 与向量。"""

        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings must align with chunks")
        with self._lock:
            # 先建立旧索引里 chunk_id -> 下标的映射，方便做 O(1) 替换。
            id_to_idx = {c.metadata.chunk_id: i for i, c in enumerate(self._chunks)}
            for ch, emb in zip(chunks, embeddings, strict=True):
                cid = ch.metadata.chunk_id
                if cid in id_to_idx:
                    idx = id_to_idx[cid]
                    self._chunks[idx] = ch
                    if self._embeddings is not None and self._embeddings.shape[0] > idx:
                        # 如果是更新已有 chunk，就原位替换对应向量。
                        self._embeddings[idx] = emb
                else:
                    id_to_idx[cid] = len(self._chunks)
                    self._chunks.append(ch)
                    if self._embeddings is None:
                        # 首个向量需要单独构造二维矩阵。
                        self._embeddings = np.stack([emb], axis=0)
                    else:
                        # 其余情况按行追加。
                        self._embeddings = np.vstack([self._embeddings, emb.reshape(1, -1)])

    def replace_all(self, chunks: list[TextChunk], embeddings: np.ndarray | None) -> None:
        """整体替换索引内容。适合全量重建。"""

        with self._lock:
            self._chunks = list(chunks)
            self._embeddings = embeddings

    def get_all_chunks(self) -> list[TextChunk]:
        """返回 chunk 列表副本，避免外部直接改写内部状态。"""

        with self._lock:
            return list(self._chunks)

    def get_embeddings(self) -> np.ndarray | None:
        """返回向量矩阵副本。"""

        with self._lock:
            return self._embeddings.copy() if self._embeddings is not None else None

    @property
    def chunk_count(self) -> int:
        """当前索引中的 chunk 数量。"""

        with self._lock:
            return len(self._chunks)
