"""链路追踪配置模块。

当前目标不是做复杂 tracing 编排，而是给项目留一个统一的 OTel 接入口：
- 本地不配 exporter 也能正常启动
- 配了 OTLP endpoint 后可以平滑接到外部 tracing 系统
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from core.config.settings import get_settings


def setup_tracing() -> trace.Tracer:
    """初始化 tracer 并按配置接入 OTLP 导出。"""

    settings = get_settings()
    resource = Resource.create(
        {"service.name": settings.otel_service_name, "deployment.environment": settings.app_env}
    )
    provider = TracerProvider(resource=resource)
    if settings.otel_exporter_otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(
                endpoint=settings.otel_exporter_otlp_endpoint, insecure=True
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except Exception:  # noqa: BLE001
            # tracing 不能反过来把主业务链路拖挂，所以这里静默降级。
            pass
    trace.set_tracer_provider(provider)
    return trace.get_tracer(settings.otel_service_name)
