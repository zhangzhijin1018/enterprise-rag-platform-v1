"""健康检查路由模块。用于提供服务可用性探针，方便本地、容器和 K8s 场景接入。"""

from fastapi import APIRouter

from apps.api.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")
