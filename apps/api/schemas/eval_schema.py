"""评测接口数据模型模块。定义评测任务的返回结构与摘要字段。"""

from pathlib import Path

from pydantic import BaseModel, Field


class EvalRequest(BaseModel):
    dataset_path: Path | None = Field(
        default=None, description="Optional override; defaults to settings.eval_dataset_path"
    )


class EvalResponse(BaseModel):
    status: str = "completed"
    report_path: str
    summary: dict[str, float] | None = None
