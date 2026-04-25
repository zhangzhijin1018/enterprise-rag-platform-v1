"""入库流水线模块。

这里把“文件解析 -> 元数据补全 -> 语义切块 -> 向量化 -> 持久化 -> 检索器重载”
串成一个可复用的最短路径，是理解知识接入流程的核心入口之一。

如果你想看“一个文件最终怎么变成可检索知识”的最短主线，
这就是最值得先读的文件之一。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.ingestion.chunkers.semantic_chunker import SemanticChunker
from core.ingestion.metadata_extractors.basic import BasicMetadataExtractor
from core.ingestion.parsers.registry import get_parser_for_filename
from core.models.document import Document, TextChunk
from core.retrieval.dense_retriever import DenseRetriever
from core.retrieval.milvus_retriever import MilvusDenseRetriever
from core.services.runtime import RAGRuntime


def parse_and_chunk_file(path: Path, source: str | None = None) -> tuple[Document, list[TextChunk]]:
    """解析单个文件并完成切块。

    返回值：
    - `Document`：标准化后的整篇文档对象。
    - `list[TextChunk]`：可直接进入检索索引的 chunk 列表。

    这一步只负责把“文件内容”变成“可索引内容”，还不负责真正写入向量库。
    这样做的好处是：
    - 更容易单测
    - 更容易在入库前人工检查中间结果
    - 后续如果要做 dry-run / 预览，也能直接复用
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
    # 元数据提取器负责把“原始文件”提升成“带企业语义的知识对象”。
    # 这一步会补：
    # - 文档身份字段
    # - 组织归属
    # - 业务域 / 流程阶段
    # - 数据分级 / 权威级别
    meta_ex = BasicMetadataExtractor()
    doc = meta_ex.ensure_doc_id(doc)
    doc = meta_ex.infer_title_from_filename(path, doc)
    doc = meta_ex.enrich_retrieval_metadata(path, doc)
    # 语义切块器进一步把文档转成 parent + child 两层 chunk：
    # - child 更适合精准召回
    # - parent 更适合回扩和生成
    chunker = SemanticChunker()
    chunks = chunker.chunk(doc)
    return doc, chunks


def index_chunks(
    runtime: RAGRuntime,
    chunks: list[TextChunk],
    *,
    replace_all: bool = False,
) -> None:
    """把 chunks 写入索引，并立即刷新检索器。

    当前这一轮开始，Milvus 作为唯一权威存储：
    - 入库后不再写本地 `chunks.jsonl / embeddings.npy`
    - 而是直接全量同步到 Milvus
    - reload 时再从 Milvus 重建 sparse / dense 的内存视图

    所以这个函数本质上做的是两件事：
    1. 重新计算这一批 chunk 的 dense 向量
    2. 把最新快照同步进 Milvus，并让运行时切到新索引
    """

    # 这里临时新建 DenseRetriever，只负责重新编码文档向量。
    # 真正供线上查询使用的 `runtime.dense` 会在最后 `reload_index()` 时统一更新。
    dense = DenseRetriever(runtime.settings)
    all_chunks = list(chunks)
    if not replace_all:
        # 增量模式下，旧 chunk 会和新 chunk 合并；chunk_id 相同的记录会被新值覆盖。
        existing = runtime.dense.fetch_all_chunks() if isinstance(runtime.dense, MilvusDenseRetriever) else []
        id_new = {c.metadata.chunk_id for c in chunks}
        merged = [c for c in existing if c.metadata.chunk_id not in id_new] + all_chunks
        all_chunks = merged
    # 向量模型输入的是纯文本列表，所以这里只取 chunk.content。
    texts = [c.content for c in all_chunks]
    emb = dense.embed_documents(texts)
    # 当前统一把全量快照同步到 Milvus。
    runtime.dense.sync_remote_index(all_chunks, np.asarray(emb))
    # Milvus 更新后，必须刷新运行时检索器，否则线上查询仍会读旧内存。
    runtime.reload_index()


def rebuild_index_from_milvus(runtime: RAGRuntime) -> None:
    """根据 Milvus 中已有的 chunk 文本重新生成 embeddings。

    适合：
    - embedding 模型切换
    - 检索器重建
    - 不想重新解析原始文件，只想重算 dense 向量

    它和 `parse_and_chunk_file()` 的区别是：
    - 这里不重新读原文件
    - 不重做 parser / metadata / chunking
    - 只基于 Milvus 中已有 chunk 文本重建向量层
    """

    chunks = runtime.dense.fetch_all_chunks() if isinstance(runtime.dense, MilvusDenseRetriever) else []
    if not chunks:
        # 没有 chunk 时直接把运行时刷新为空索引即可。
        runtime.reload_index()
        return
    dense = DenseRetriever(runtime.settings)
    texts = [c.content for c in chunks]
    emb = dense.embed_documents(texts)
    # 这里不会改 chunk 内容，只是重算 dense 向量并整体回写 Milvus。
    runtime.dense.sync_remote_index(chunks, np.asarray(emb))
    runtime.reload_index()
