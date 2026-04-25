"""FastAPI 应用入口。

这个文件把路由、CORS、中间件、Prometheus 指标和静态前端统一装配成一个服务。
学习时可以把它当成“整个项目从 HTTP 入口进入的总开关”。
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from apps.api.routes import chat, eval as eval_routes, faq, health, ingest
from core.config.settings import get_settings
from core.observability import (
    clear_request_log_context,
    configure_logging,
    get_logger,
    set_request_log_context,
)
from core.observability.metrics import REQUEST_LATENCY, metrics_response
from core.observability.tracing import setup_tracing

logger = get_logger(__name__)

# 如果前端已经执行过构建，这里会指向打包后的静态资源目录。
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期钩子。

    启动阶段做三件事：
    1. 初始化日志。
    2. 初始化链路追踪。
    3. 尝试给 FastAPI 自动打 OpenTelemetry 埋点。
    """

    settings = get_settings()
    configure_logging(
        settings.log_level,
        enable_file_logging=settings.enable_file_logging,
        log_dir=settings.log_dir,
        app_log_filename=settings.app_log_filename,
        audit_log_filename=settings.audit_log_filename,
        log_max_bytes=settings.log_max_bytes,
        log_backup_count=settings.log_backup_count,
    )
    setup_tracing()
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        # 自动给每个请求加 trace/span，后续在 OTEL 后端里能看到完整请求链路。
        FastAPIInstrumentor.instrument_app(app)
    except Exception as e:  # noqa: BLE001
        # 这里选择降级而不是让服务启动失败，便于本地无 OTEL 依赖时继续开发。
        logger.warning("FastAPI OpenTelemetry instrumentation skipped: %s", e)
    yield


# 把 lifespan 挂给应用后，FastAPI 会在启动和关闭阶段自动调用。
app = FastAPI(title="Enterprise RAG Platform", lifespan=lifespan)

_settings = get_settings()
# 允许通过逗号分隔配置多个前端来源。
_origins = [o.strip() for o in _settings.cors_origins.split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(faq.router)
app.include_router(ingest.router)
app.include_router(eval_routes.router)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """记录请求耗时。

    原理：
    - 在请求进入时记录高精度时间戳。
    - 等待下游路由执行完成。
    - 把耗时写入 Prometheus Histogram，后续可以按路由统计延迟分布。
    """

    trace_id = request.headers.get("X-Trace-ID") or uuid4().hex
    request.state.trace_id = trace_id
    request.state.request_started_at = time.time()
    set_request_log_context(
        trace_id=trace_id,
        event="request_received",
        request_path=request.url.path,
        method=request.method,
    )
    t0 = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        return response
    finally:
        path = request.url.path
        REQUEST_LATENCY.labels(route=path, method=request.method).observe(time.perf_counter() - t0)
        clear_request_log_context()


@app.get("/metrics")
def metrics():
    """暴露 Prometheus 文本格式指标。"""

    data, ctype = metrics_response()
    return Response(content=data, media_type=ctype)


@app.get("/")
def root():
    """根路径入口。

    - 如果前端构建产物存在，就跳到 `/ui/`。
    - 否则返回一个简单 JSON，提醒开发者先构建前端。
    """

    if WEB_DIST.is_dir():
        return RedirectResponse(url="/ui/")
    return {
        "service": "enterprise-rag-api",
        "docs": "/docs",
        "ui": "build apps/web (npm run build) to enable /ui/",
    }


if WEB_DIST.is_dir():
    # 这里把构建后的 React 应用挂到 `/ui`，实现“一个进程同时服务 API 和静态前端”。
    app.mount(
        "/ui",
        StaticFiles(directory=str(WEB_DIST), html=True),
        name="ui",
    )
