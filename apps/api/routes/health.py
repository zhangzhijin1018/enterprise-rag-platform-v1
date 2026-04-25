"""健康检查路由模块。用于提供服务可用性探针，方便本地、容器和 K8s 场景接入。"""

from fastapi import APIRouter

from apps.api.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """返回最小健康检查结果。

    当前只做进程级健康探针，不在这里探测所有外部依赖，
    这样健康检查更稳定，也更适合容器 / K8s 的 liveness probe。
    """

    return HealthResponse(status="ok")
