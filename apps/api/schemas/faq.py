"""FAQ 管理接口的数据模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class FaqImportResponse(BaseModel):
    imported: int
    status: str


class FaqItemSchema(BaseModel):
    id: int
    question: str
    answer: str
    keywords: str = ""
    category: str = ""
    enabled: bool
    hit_count: int = 0
    last_hit_at: datetime | None = None


class FaqListResponse(BaseModel):
    items: list[FaqItemSchema]


class FaqToggleRequest(BaseModel):
    enabled: bool


class FaqToggleResponse(BaseModel):
    id: int
    enabled: bool
    status: str


class FaqUpdateRequest(BaseModel):
    question: str
    answer: str
    keywords: str = ""
    category: str = ""


class FaqUpdateResponse(BaseModel):
    id: int
    status: str
