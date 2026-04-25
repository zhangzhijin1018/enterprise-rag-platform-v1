"""MySQL FAQ 存储模块。

这个模块负责把“结构化 FAQ 库”落到 MySQL。

为什么要单独做这层，而不是把 FAQ 继续混在普通 RAG 文档里：

1. FAQ 往往是一问一答结构，适合先走快速命中
2. 结构化表更适合做运营维护、人工导入和后续后台管理
3. 命中 FAQ 后可以直接返回答案，不必再走完整 RAG 链路

因此，这层在整体架构中的定位是：

- MySQL：结构化 FAQ 的持久化来源
- BM25 FAQ 检索器：把 MySQL 中的 FAQ 加载到内存后做轻量检索
- Redis：缓存热点问题的最终答案
- RAG：当 FAQ 没命中时再兜底
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.config.settings import Settings, get_settings
from core.observability import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class FaqEntry:
    """FAQ 结构化记录。"""

    entry_id: int
    question: str
    answer: str
    keywords: str = ""
    category: str = ""
    enabled: bool = True
    hit_count: int = 0
    last_hit_at: datetime | None = None


class MysqlFaqStore:
    """MySQL FAQ 存储访问层。

    这层只做 FAQ 持久化和基础管理，不负责检索排序。
    检索排序交给 `MysqlFaqRetriever` 在内存里完成。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._engine = None
        self._enabled = bool(self._settings.mysql_url)

    @property
    def enabled(self) -> bool:
        """当前 MySQL FAQ 存储是否启用。"""

        return self._enabled

    def _get_engine(self):
        """延迟创建 SQLAlchemy engine。

        这里延迟导入的目的是：

        - 测试环境不需要时，不强制建立数据库连接
        - 启动阶段如果 MySQL 不可用，可以记录告警并降级，而不是直接把服务打崩
        """

        if not self._enabled:
            return None
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
            except ModuleNotFoundError as exc:  # pragma: no cover - 环境依赖问题
                raise RuntimeError(
                    "MySQL FAQ store requires `sqlalchemy` and `pymysql`."
                ) from exc
            self._engine = create_engine(
                self._settings.mysql_url,
                pool_pre_ping=True,
                future=True,
            )
        return self._engine

    def initialize(self) -> None:
        """初始化 FAQ 表，并在空表时写入示例数据。"""

        if not self._enabled:
            return
        try:
            from sqlalchemy import text

            engine = self._get_engine()
            if engine is None:
                return
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS faq_entries (
                            id BIGINT PRIMARY KEY AUTO_INCREMENT,
                            question VARCHAR(512) NOT NULL UNIQUE,
                            answer TEXT NOT NULL,
                            keywords VARCHAR(1024) NOT NULL DEFAULT '',
                            category VARCHAR(128) NOT NULL DEFAULT '',
                            enabled TINYINT(1) NOT NULL DEFAULT 1,
                            hit_count BIGINT NOT NULL DEFAULT 0,
                            last_hit_at TIMESTAMP NULL DEFAULT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
                        """
                    )
                )
                # 这些列是第七轮增强新增的 FAQ 管理字段。
                # 使用 `IF NOT EXISTS` 的好处是：
                # - 老数据库可以平滑升级
                # - 新数据库也不会重复报错
                conn.execute(
                    text(
                        "ALTER TABLE faq_entries "
                        "ADD COLUMN IF NOT EXISTS hit_count BIGINT NOT NULL DEFAULT 0"
                    )
                )
                conn.execute(
                    text(
                        "ALTER TABLE faq_entries "
                        "ADD COLUMN IF NOT EXISTS last_hit_at TIMESTAMP NULL DEFAULT NULL"
                    )
                )
            self.seed_if_empty(Path(self._settings.faq_seed_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("mysql faq store unavailable, fast path degraded: %s", exc)
            self._enabled = False

    def seed_if_empty(self, seed_path: Path) -> None:
        """当 FAQ 表为空时，用本地种子 CSV 初始化。

        这样项目在第一次启动时就有一批可演示、可测试的 FAQ 数据，
        不需要手工先往 MySQL 里插数据。
        """

        if not self._enabled or not seed_path.is_file():
            return
        try:
            from sqlalchemy import text

            engine = self._get_engine()
            if engine is None:
                return
            with engine.begin() as conn:
                count = conn.execute(text("SELECT COUNT(*) AS total FROM faq_entries")).scalar_one()
                if int(count or 0) > 0:
                    return
            entries = self._read_csv(seed_path)
            if entries:
                self.upsert_entries(entries)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mysql faq seed skipped: %s", exc)

    def _read_csv(self, path: Path) -> list[FaqEntry]:
        """把 CSV 文件解析为 FAQ 列表。"""

        rows: list[FaqEntry] = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                question = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()
                if not question or not answer:
                    continue
                rows.append(
                    FaqEntry(
                        entry_id=int(row.get("id") or idx),
                        question=question,
                        answer=answer,
                        keywords=(row.get("keywords") or "").strip(),
                        category=(row.get("category") or "").strip(),
                        enabled=True,
                        hit_count=0,
                        last_hit_at=None,
                    )
                )
        return rows

    def import_csv(self, path: Path) -> int:
        """从 CSV 导入 FAQ，并返回导入条数。

        当前导入策略偏简单直接：
        - 先全量读 CSV
        - 再按 question 做 upsert
        """

        entries = self._read_csv(path)
        self.upsert_entries(entries)
        return len(entries)

    def upsert_entries(self, entries: list[FaqEntry]) -> None:
        """把 FAQ 批量写入 MySQL。

        这里按 `question` 做唯一约束，原因是：

        - FAQ 业务上通常也是按“问题”作为主键语义维护
        - 这样从 CSV 反复导入时，可以做到幂等更新
        """

        if not self._enabled or not entries:
            return
        from sqlalchemy import text

        engine = self._get_engine()
        if engine is None:
            return
        with engine.begin() as conn:
            for entry in entries:
                conn.execute(
                    text(
                        """
                        INSERT INTO faq_entries (question, answer, keywords, category, enabled)
                        VALUES (:question, :answer, :keywords, :category, 1)
                        ON DUPLICATE KEY UPDATE
                            answer = VALUES(answer),
                            keywords = VALUES(keywords),
                            category = VALUES(category),
                            enabled = 1
                        """
                    ),
                    {
                        "question": entry.question,
                        "answer": entry.answer,
                        "keywords": entry.keywords,
                        "category": entry.category,
                    },
                )

    def list_enabled_entries(self) -> list[FaqEntry]:
        """读取当前启用的 FAQ 条目。

        这批结果后续会被 FAQ 检索器整体读入内存，重建 BM25 索引。
        """

        if not self._enabled:
            return []
        try:
            from sqlalchemy import text

            engine = self._get_engine()
            if engine is None:
                return []
            with engine.begin() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT id, question, answer, keywords, category, enabled, hit_count, last_hit_at
                        FROM faq_entries
                        WHERE enabled = 1
                        ORDER BY id ASC
                        """
                    )
                )
                out: list[FaqEntry] = []
                for row in rows.mappings():
                    out.append(
                        FaqEntry(
                            entry_id=int(row["id"]),
                            question=str(row["question"] or ""),
                            answer=str(row["answer"] or ""),
                            keywords=str(row["keywords"] or ""),
                            category=str(row["category"] or ""),
                            enabled=bool(row["enabled"]),
                            hit_count=int(row["hit_count"] or 0),
                            last_hit_at=row["last_hit_at"],
                        )
                    )
                return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("mysql faq list failed: %s", exc)
            self._enabled = False
            return []

    def list_all_entries(self) -> list[FaqEntry]:
        """读取 FAQ 全量条目，给管理页展示用。"""

        if not self._enabled:
            return []
        try:
            from sqlalchemy import text

            engine = self._get_engine()
            if engine is None:
                return []
            with engine.begin() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT id, question, answer, keywords, category, enabled, hit_count, last_hit_at
                        FROM faq_entries
                        ORDER BY id ASC
                        """
                    )
                )
                out: list[FaqEntry] = []
                for row in rows.mappings():
                    out.append(
                        FaqEntry(
                            entry_id=int(row["id"]),
                            question=str(row["question"] or ""),
                            answer=str(row["answer"] or ""),
                            keywords=str(row["keywords"] or ""),
                            category=str(row["category"] or ""),
                            enabled=bool(row["enabled"]),
                            hit_count=int(row["hit_count"] or 0),
                            last_hit_at=row["last_hit_at"],
                        )
                    )
                return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("mysql faq list failed: %s", exc)
            self._enabled = False
            return []

    def set_enabled(self, entry_id: int, enabled: bool) -> bool:
        """启用或停用 FAQ 条目。"""

        if not self._enabled:
            return False
        try:
            from sqlalchemy import text

            engine = self._get_engine()
            if engine is None:
                return False
            with engine.begin() as conn:
                result = conn.execute(
                    text(
                        """
                        UPDATE faq_entries
                        SET enabled = :enabled
                        WHERE id = :entry_id
                        """
                    ),
                    {"enabled": 1 if enabled else 0, "entry_id": entry_id},
                )
                return int(result.rowcount or 0) > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("mysql faq enable toggle failed: %s", exc)
            self._enabled = False
            return False

    def update_entry(
        self,
        entry_id: int,
        *,
        question: str,
        answer: str,
        keywords: str,
        category: str,
    ) -> bool:
        """更新 FAQ 主内容。

        当前有意不支持在线删除，原因很实际：

        - 启停已经能满足绝大多数运营场景
        - 删除更容易误操作
        - 先把编辑路径做稳，再考虑更强管理能力
        """

        if not self._enabled:
            return False
        try:
            from sqlalchemy import text

            engine = self._get_engine()
            if engine is None:
                return False
            with engine.begin() as conn:
                result = conn.execute(
                    text(
                        """
                        UPDATE faq_entries
                        SET question = :question,
                            answer = :answer,
                            keywords = :keywords,
                            category = :category
                        WHERE id = :entry_id
                        """
                    ),
                    {
                        "question": question,
                        "answer": answer,
                        "keywords": keywords,
                        "category": category,
                        "entry_id": entry_id,
                    },
                )
                return int(result.rowcount or 0) > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("mysql faq update failed: %s", exc)
            self._enabled = False
            return False

    def record_hit(self, entry_id: int) -> None:
        """记录 FAQ 被快速通道命中的次数和时间。"""

        if not self._enabled:
            return
        try:
            from sqlalchemy import text

            engine = self._get_engine()
            if engine is None:
                return
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE faq_entries
                        SET hit_count = hit_count + 1,
                            last_hit_at = CURRENT_TIMESTAMP
                        WHERE id = :entry_id
                        """
                    ),
                    {"entry_id": entry_id},
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("mysql faq hit record failed: %s", exc)
