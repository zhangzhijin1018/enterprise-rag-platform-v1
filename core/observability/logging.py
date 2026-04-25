"""日志配置模块。

职责分成两层：
1. 给 root logger 配置 stdout 与本地 `.log` 文件输出；
2. 通过 `contextvars` 维护请求级日志上下文，让链路中的每条日志都能自动带上
   `trace_id / audit_id / user_id / department / role / event` 等关键字段。
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})
_STANDARD_RECORD_ATTRS = set(logging.makeLogRecord({}).__dict__.keys())


def get_request_log_context() -> dict[str, Any]:
    """获取当前协程的日志上下文。"""

    return dict(_LOG_CONTEXT.get())


def set_request_log_context(**values: Any) -> None:
    """覆盖设置当前请求的日志上下文。"""

    _LOG_CONTEXT.set({key: value for key, value in values.items() if value not in (None, "", [], {})})


def update_request_log_context(**values: Any) -> None:
    """增量更新当前请求的日志上下文。"""

    current = dict(_LOG_CONTEXT.get())
    for key, value in values.items():
        if value in (None, "", [], {}):
            continue
        current[key] = value
    _LOG_CONTEXT.set(current)


def clear_request_log_context() -> None:
    """清空当前请求的日志上下文。"""

    _LOG_CONTEXT.set({})


class RequestContextFilter(logging.Filter):
    """把请求级上下文字段注入到每条日志记录。

    这样业务代码只需要正常 `logger.info(...)`，格式化器就能自动拿到：
    - trace_id
    - audit_id
    - user_id / department / role
    - event
    - data_classification / model_route
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        context = get_request_log_context()
        for key, value in context.items():
            if getattr(record, key, None) in (None, "", [], {}):
                setattr(record, key, value)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_ATTRS or key in {"message", "asctime"}:
                continue
            if value in (None, "", [], {}):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class LineFormatter(logging.Formatter):
    """面向本地排障的 `.log` 文本格式。

    输出风格尽量接近传统应用日志，便于：
    - 人眼直接 grep / less
    - 按 `trace_id` 回放一次请求走过的关键步骤
    """

    _FIELD_ORDER = (
        "trace_id",
        "audit_id",
        "event",
        "user_id",
        "department",
        "role",
        "project_ids",
        "data_classification",
        "model_route",
    )

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        base = f"{ts} {record.levelname} [{record.name}]"
        extras: list[str] = []
        used_keys: set[str] = set()
        for key in self._FIELD_ORDER:
            value = getattr(record, key, None)
            if value in (None, "", [], {}):
                continue
            extras.append(f"{key}={value}")
            used_keys.add(key)
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_ATTRS or key in {"message", "asctime"} or key in used_keys:
                continue
            if value in (None, "", [], {}):
                continue
            extras.append(f"{key}={value}")
        line = f"{base} {' '.join(extras)} message={record.getMessage()}".rstrip()
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def _build_rotating_handler(
    path: Path,
    level: str,
    formatter: logging.Formatter,
    *,
    max_bytes: int,
    backup_count: int,
) -> RotatingFileHandler:
    """创建轮转文件日志 handler。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        filename=path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level.upper())
    handler.setFormatter(formatter)
    handler.addFilter(RequestContextFilter())
    return handler


def configure_logging(
    level: str = "INFO",
    *,
    enable_file_logging: bool = True,
    log_dir: str = "./logs",
    app_log_filename: str = "app.log",
    audit_log_filename: str = "audit.log",
    log_max_bytes: int = 10 * 1024 * 1024,
    log_backup_count: int = 5,
) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.filters.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level.upper())
    stream_handler.setFormatter(JsonFormatter())
    stream_handler.addFilter(RequestContextFilter())
    root.addHandler(stream_handler)
    root.setLevel(level.upper())

    audit_logger = logging.getLogger("audit")
    audit_logger.handlers.clear()
    audit_logger.setLevel(level.upper())
    audit_logger.propagate = True

    if enable_file_logging:
        log_path = Path(log_dir)
        line_formatter = LineFormatter()
        root.addHandler(
            _build_rotating_handler(
                log_path / app_log_filename,
                level,
                line_formatter,
                max_bytes=log_max_bytes,
                backup_count=log_backup_count,
            )
        )
        audit_logger.addHandler(
            _build_rotating_handler(
                log_path / audit_log_filename,
                level,
                line_formatter,
                max_bytes=log_max_bytes,
                backup_count=log_backup_count,
            )
        )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
