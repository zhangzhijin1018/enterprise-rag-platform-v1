"""FAQ 检索单元测试。"""

from core.config.settings import Settings
from core.retrieval.faq_retriever import MysqlFaqRetriever
from core.retrieval.faq_store import FaqEntry


def test_mysql_faq_retriever_ranks_matching_question_first() -> None:
    settings = Settings(FAQ_TOP_K=2)
    retriever = MysqlFaqRetriever(settings)
    retriever.rebuild(
        [
            FaqEntry(
                entry_id=1,
                question="错误码 E-1001 是什么？",
                answer="表示 Redis connection failed。",
                keywords="E-1001,Redis",
                category="error_code",
            ),
            FaqEntry(
                entry_id=2,
                question="Milvus 和 Zilliz Cloud 有什么区别？",
                answer="一个开源一个托管。",
                keywords="Milvus,Zilliz Cloud,对比",
                category="product_compare",
            ),
        ]
    )

    hits = retriever.search("E-1001 这个错误码是什么意思", top_k=2)

    assert hits
    assert hits[0].entry.entry_id == 1
    assert hits[0].confidence > 0.5
