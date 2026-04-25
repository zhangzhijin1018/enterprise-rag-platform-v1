"""API 通用数据模型模块。放置多个接口共用的响应结构或状态模型。"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    """服务健康状态。"""


class IngestResponse(BaseModel):
    job_id: str
    """后台任务 ID。"""
    status: str = "accepted"
    """任务接收状态。"""


class JobStatusResponse(BaseModel):
    job_id: str
    """后台任务 ID。"""
    status: str
    """后台任务当前状态。"""
    detail: str | None = None
    """失败原因或附加说明。"""
