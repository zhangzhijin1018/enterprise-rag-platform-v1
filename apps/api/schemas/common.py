"""API 通用数据模型模块。放置多个接口共用的响应结构或状态模型。"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"


class IngestResponse(BaseModel):
    job_id: str
    status: str = "accepted"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    detail: str | None = None
