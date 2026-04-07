"""入库流水线模块。

这里把“文件解析 -> 元数据补全 -> 语义切块 -> 向量化 -> 持久化 -> 检索器重载”
串成一个可复用的最短路径，是理解知识接入流程的核心入口之一。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.ingestion.chunkers.semantic_chunker import SemanticChunker
from core.ingestion.metadata_extractors.basic import BasicMetadataExtractor
from core.ingestion.parsers.registry import get_parser_for_filename
from core.models.document import Document, TextChunk
from core.retrieval.dense_retriever import DenseRetriever
from core.retrieval.index_store import IndexStore
from core.services.runtime import RAGRuntime


def parse_and_chunk_file(path: Path, source: str | None = None) -> tuple[Document, list[TextChunk]]:
    """解析单个文件并完成切块。

    返回值：
    - `Document`：标准化后的整篇文档对象。
    - `list[TextChunk]`：可直接进入检索索引的 chunk 列表。
    """

    # `source` 用于记录文档来源；如果外部没传，就退化为文件路径字符串。
    src = source or str(path)
    # 先根据文件扩展名选择解析器。
    #
    # 当前已经支持：
    # - PDF
    # - DOCX
    # - HTML
    # - Markdown
    # - TXT
    # - CSV
    # - PPTX
    #
    # 这里刻意把“格式判断”集中放在 registry，而不是散落在 pipeline 中，
    # 这样后续扩展新文件类型时，主流程本身不需要改动。
    parser = get_parser_for_filename(path.name)
    # 解析器负责把原始文件统一成 Document。
    doc = parser.parse(path, src)
    # 元数据提取器补齐 doc_id、标题等后续检索与引用要依赖的字段。
    meta_ex = BasicMetadataExtractor()
    doc = meta_ex.ensure_doc_id(doc)
    doc = meta_ex.infer_title_from_filename(path, doc)
    # 语义切块器把长文拆成多个较短片段，兼顾召回精度与上下文长度限制。
    chunker = SemanticChunker()
    chunks = chunker.chunk(doc)
    return doc, chunks


def index_chunks(
    runtime: RAGRuntime,
    chunks: list[TextChunk],
    *,
    replace_all: bool = False,
) -> None:
    """把 chunks 写入索引，并立即刷新检索器。"""

    store: IndexStore = runtime.store
    # 这里临时新建 DenseRetriever，只负责重新编码文档向量。
    # 真正供线上查询使用的 `runtime.dense` 会在最后 `reload_index()` 时统一更新。
    dense = DenseRetriever(runtime.settings)
    all_chunks = list(chunks)
    if not replace_all:
        # 增量模式下，旧 chunk 会和新 chunk 合并；chunk_id 相同的记录会被新值覆盖。
        existing = store.get_all_chunks()
        id_new = {c.metadata.chunk_id for c in chunks}
        merged = [c for c in existing if c.metadata.chunk_id not in id_new] + all_chunks
        all_chunks = merged
    # 向量模型输入的是纯文本列表，所以这里只取 chunk.content。
    texts = [c.content for c in all_chunks]
    emb = dense.embed_documents(texts)
    # `replace_all` 的好处是磁盘上的 chunks 和 embeddings 总能保持完全对齐。
    store.replace_all(all_chunks, np.asarray(emb))
    store.save()
    # 如果当前启用了 Milvus backend，这里会把最新全量快照同步到 Milvus。
    # 文件型 backend 下该方法是 no-op，不会引入额外成本。
    runtime.dense.sync_remote_index(all_chunks, np.asarray(emb))
    # 索引文件落盘后，必须刷新运行时检索器，否则线上查询仍会读旧内存。
    runtime.reload_index()


def rebuild_index_from_store_files(runtime: RAGRuntime) -> None:
    """根据磁盘上的 `chunks.jsonl` 重新生成 embeddings。"""

    # 先把当前磁盘状态读到内存。
    runtime.store.load()
    chunks = runtime.store.get_all_chunks()
    if not chunks:
        # 没有 chunks 时直接把运行时刷新为空索引即可。
        runtime.reload_index()
        return
    dense = DenseRetriever(runtime.settings)
    texts = [c.content for c in chunks]
    emb = dense.embed_documents(texts)
    # 这里不会改 chunk 内容，只是重算向量矩阵，适合模型切换后的全量重建。
    runtime.store.replace_all(chunks, np.asarray(emb))
    runtime.store.save()
    runtime.dense.sync_remote_index(chunks, np.asarray(emb))
    runtime.reload_index()
