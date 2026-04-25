"""评测接口数据模型模块。定义评测任务的返回结构与摘要字段。"""

from pathlib import Path

from pydantic import BaseModel, Field


class EvalRequest(BaseModel):
    dataset_path: Path | None = Field(
        default=None, description="评测数据集路径；为空时默认使用 settings.eval_dataset_path"
    )


class EvalResponse(BaseModel):
    status: str = "completed"
    """评测任务状态。"""
    report_path: str
    """JSON 评测报告路径。"""
    analysis_path: str | None = None
    """Markdown explainability 报告路径。"""
    summary: dict[str, float] | None = None
    """评测摘要指标。"""
