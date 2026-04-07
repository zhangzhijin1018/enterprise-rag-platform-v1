"""运行时依赖装配模块。

`RAGRuntime` 的作用类似一个轻量级 service container：
它把索引存储、检索器、LLM、缓存和 LangGraph 编排实例放在一起，
让 API 与后台任务能共享同一套可复用依赖。
"""

from __future__ import annotations

import threading
from typing import Any

from core.config.settings import get_settings
from core.generation.llm_client import LLMClient
from core.retrieval.faq_retriever import MysqlFaqRetriever
from core.retrieval.faq_store import MysqlFaqStore
from core.retrieval.cache import RedisCache
from core.retrieval.dense_retriever import DenseRetriever
from core.retrieval.hybrid_fusion import HybridFusion
from core.retrieval.index_store import IndexStore
from core.retrieval.milvus_retriever import MilvusDenseRetriever
from core.retrieval.reranker import CrossEncoderReranker
from core.retrieval.sparse_retriever import SparseRetriever


class RAGRuntime:
    """装配 RAG 主链路运行时所需的全部依赖。"""

    def __init__(self) -> None:
        # 先读取全局配置，后续所有依赖都基于这份配置初始化。
        self.settings = get_settings()
        # `IndexStore` 负责管理 chunks 和 embeddings 的磁盘持久化。
        self.store = IndexStore(self.settings)
        # 稀疏检索器通常初始化便宜，所以直接常驻。
        self.sparse = SparseRetriever(self.settings)
        # 稠密检索器和 reranker 模型较重，采用懒加载，减少冷启动成本。
        #
        # 第五轮增强后，这里会根据 `VECTOR_BACKEND` 选择：
        # - `file`   -> 本地 NumPy 向量矩阵
        # - `milvus` -> Milvus / Milvus Lite
        self._dense: DenseRetriever | None = None
        self.fusion = HybridFusion(self.settings)
        self._reranker: CrossEncoderReranker | None = None
        self.llm = LLMClient(self.settings)
        self.cache = RedisCache()
        self.faq_store = MysqlFaqStore(self.settings)
        self.faq_retriever = MysqlFaqRetriever(self.settings)
        # 编译后的 LangGraph 可以复用，避免每次问答都重新构图。
        self._compiled_graph: Any = None
        self.reload_faq_index()
        # 启动时先加载已有索引，这样服务一起来就能直接问答。
        self.reload_index()

    @property
    def dense(self) -> DenseRetriever:
        """按需初始化向量检索器。"""

        if self._dense is None:
            if self.settings.vector_backend == "milvus":
                self._dense = MilvusDenseRetriever(self.settings)
            else:
                self._dense = DenseRetriever(self.settings)
        return self._dense

    @property
    def reranker(self) -> CrossEncoderReranker:
        """按需初始化交叉编码器重排器。"""

        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.settings)
        return self._reranker

    def get_compiled_graph(self) -> Any:
        """返回已编译的 LangGraph。

        为什么要缓存：
        - 构图和 compile 都有固定开销。
        - 图结构不会随着单次请求变化，所以很适合复用。
        """

        if self._compiled_graph is None:
            from core.orchestration.graph import build_rag_graph

            self._compiled_graph = build_rag_graph(self).compile()
        return self._compiled_graph

    def reload_index(self) -> None:
        """重新加载磁盘索引，并同步刷新检索器状态。"""

        # 第一步：把磁盘里的 chunks / embeddings 读回内存。
        self.store.load()
        chunks = self.store.get_all_chunks()
        emb = self.store.get_embeddings()
        # 第二步：重建 BM25 词项统计。
        self.sparse.rebuild(chunks)
        if chunks:
            # 第三步：如果已有向量，就直接喂给向量检索器，避免重复编码。
            self.dense.rebuild(chunks, emb)
            # 如果当前使用的是 Milvus backend，而远端 collection 还不存在，
            # 这里会利用本地镜像自动补建一次，保证“已有索引快照 -> 服务重启”
            # 这种场景下仍能正常工作。
            self.dense.ensure_remote_index(chunks, emb)
        else:
            # 索引为空时也要把向量检索器重置，避免保留旧状态。
            self.dense.rebuild([], None)
        # 数据变了，旧图里的运行时引用就要失效，下次访问时重新 compile。
        self._compiled_graph = None

    def reload_faq_index(self) -> None:
        """重建 MySQL FAQ 的内存检索索引。"""

        self.faq_store.initialize()
        entries = self.faq_store.list_enabled_entries()
        self.faq_retriever.rebuild(entries)


_runtime_lock = threading.Lock()
_runtime: RAGRuntime | None = None


def get_runtime() -> RAGRuntime:
    """返回全局单例运行时。

    API 路由和后台任务都通过它共享索引、模型和缓存连接。
    """

    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = RAGRuntime()
        return _runtime


def reset_runtime() -> None:
    """在测试或重载场景下清空全局运行时。"""

    global _runtime
    with _runtime_lock:
        _runtime = None
