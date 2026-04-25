"""MySQL FAQ 检索模块。

这里不直接在 MySQL 里跑全文检索，而是采用更稳的折中方案：

1. FAQ 持久化在 MySQL
2. 启动或导入后，把 FAQ 读到内存
3. 用 BM25 做轻量匹配

这么做的原因：

- 结构化维护方便
- 检索延迟低
- 实现简单、可解释、便于调试阈值
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from core.config.settings import Settings, get_settings
from core.retrieval.faq_store import FaqEntry
from core.retrieval.sparse_retriever import tokenize


_SPECIAL_ID_RE = re.compile(r"[a-z]+-\d+|[a-z]+\d+|\d{3,}", re.IGNORECASE)


@dataclass(slots=True)
class FaqMatch:
    """FAQ 命中结果。"""

    entry: FaqEntry
    bm25_score: float
    confidence: float


class MysqlFaqRetriever:
    """FAQ 内存检索器。

    它的作用是把结构化 FAQ 转成更适合快速问答的轻量召回层。
    这层的评价标准和长文 RAG 不一样，更强调“标准问句是否被问到了”。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._entries: list[FaqEntry] = []
        self._entry_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def rebuild(self, entries: Sequence[FaqEntry]) -> None:
        """根据 FAQ 条目重建 BM25 索引。

        索引文本会同时包含：

        - question
        - keywords
        - 少量 answer 文本

        这样做的目的，是在保持问题匹配为主的前提下，让答案里的关键术语也能参与召回。
        """

        self._entries = list(entries)
        self._entry_tokens = []
        for entry in self._entries:
            text = "\n".join(
                [
                    entry.question,
                    entry.keywords,
                    entry.answer[:256],
                ]
            )
            self._entry_tokens.append(tokenize(text))
        self._bm25 = BM25Okapi(self._entry_tokens) if self._entry_tokens else None

    def search(self, query: str, top_k: int | None = None) -> list[FaqMatch]:
        """执行 FAQ 检索，并返回带置信度的候选结果。

        这里采用的是“BM25 排序 + 置信度归一化”的组合策略：

        1. 用 BM25 负责排序
        2. 用 token coverage + BM25 强度共同计算 0~1 的 confidence

        这样可以兼顾：

        - BM25 的可解释性
        - 阈值判断的稳定性
        """

        k = top_k or self._settings.faq_top_k
        if not self._bm25 or not self._entries:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        out: list[FaqMatch] = []
        query_token_set = set(query_tokens)
        query_special_ids = {item.lower() for item in _SPECIAL_ID_RE.findall(query)}
        for idx, score in ranked:
            entry = self._entries[idx]
            entry_token_set = set(self._entry_tokens[idx])
            overlap = len(query_token_set & entry_token_set)
            coverage_ratio = overlap / max(1, len(query_token_set))
            bm25_strength = float(score) / (float(score) + 1.5) if score > 0 else 0.0
            # FAQ 快速通道和长文检索不同，它更强调“标准问题是否被问到了”。
            # 所以这里除了 BM25 和 token coverage，还叠加一层 question string similarity：
            #
            # - 对中文短问句、错误码问法、FAQ 标准问句改写更稳
            # - 可以弥补纯 token 检索在中文连续文本上的边界问题
            question_similarity = SequenceMatcher(
                None,
                query.strip().lower(),
                entry.question.strip().lower(),
            ).ratio()
            entry_special_ids = {
                item.lower() for item in _SPECIAL_ID_RE.findall(f"{entry.question} {entry.keywords}")
            }
            special_id_bonus = 0.0
            if query_special_ids and entry_special_ids and query_special_ids & entry_special_ids:
                # 对错误码、型号、版本号这类“强标识符”单独加分。
                # 这是 FAQ 快速通道里非常重要的业务规则：
                # 只要这类 token 精确命中，通常就意味着 FAQ 的确定性会显著提升。
                special_id_bonus = 0.35
            confidence = min(
                1.0,
                0.35 * coverage_ratio
                + 0.20 * bm25_strength
                + 0.25 * question_similarity
                + special_id_bonus,
            )
            out.append(
                FaqMatch(
                    entry=entry,
                    bm25_score=float(score),
                    confidence=confidence,
                )
            )
        return out
