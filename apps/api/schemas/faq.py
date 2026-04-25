"""FAQ 管理接口的数据模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class FaqImportResponse(BaseModel):
    imported: int
    """本次导入成功的 FAQ 条数。"""
    status: str
    """导入任务状态。"""


class FaqItemSchema(BaseModel):
    id: int
    """FAQ 主键。"""
    question: str
    """标准问题。"""
    answer: str
    """标准答案。"""
    keywords: str = ""
    """辅助检索关键词。"""
    category: str = ""
    """FAQ 分类。"""
    enabled: bool
    """是否启用。"""
    hit_count: int = 0
    """累计命中次数。"""
    last_hit_at: datetime | None = None
    """最后一次命中时间。"""


class FaqListResponse(BaseModel):
    items: list[FaqItemSchema]
    """FAQ 列表。"""


class FaqToggleRequest(BaseModel):
    enabled: bool
    """目标启停状态。"""


class FaqToggleResponse(BaseModel):
    id: int
    """FAQ 主键。"""
    enabled: bool
    """更新后的启停状态。"""
    status: str
    """操作状态。"""


class FaqUpdateRequest(BaseModel):
    question: str
    """更新后的问题文本。"""
    answer: str
    """更新后的答案文本。"""
    keywords: str = ""
    """更新后的关键词。"""
    category: str = ""
    """更新后的分类。"""


class FaqUpdateResponse(BaseModel):
    id: int
    """FAQ 主键。"""
    status: str
    """更新状态。"""
