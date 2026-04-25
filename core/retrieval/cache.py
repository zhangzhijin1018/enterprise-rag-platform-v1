"""Redis 缓存模块。

当前主要用于缓存查询改写结果，减少重复调用 LLM。
Redis 不可用时会自动降级，不影响主链路可用性。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis

from core.config.settings import get_settings
from core.observability import get_logger

logger = get_logger(__name__)


class RedisCache:
    """一个非常薄的 Redis JSON 缓存封装。

    当前主要缓存两类信息：
    - 热点最终答案
    - 查询理解 / 规划等可复用中间结果
    """

    def __init__(self) -> None:
        self._client: redis.Redis | None = None
        settings = get_settings()
        if settings.redis_url:
            try:
                self._client = redis.from_url(settings.redis_url, decode_responses=True)
                self._client.ping()
            except Exception as e:  # noqa: BLE001
                # 缓存不是核心依赖，所以这里选择告警并降级。
                logger.warning("redis unavailable, cache disabled: %s", e)
                self._client = None

    def _key(self, prefix: str, payload: str) -> str:
        """构造稳定缓存键。

        这里不直接用原始 payload 做 key，是为了：
        - 避免 key 过长
        - 避免日志和 Redis key 里直接暴露原始问题文本
        """

        h = hashlib.sha256(payload.encode()).hexdigest()[:24]
        return f"erp:{prefix}:{h}"

    def get_json(self, prefix: str, payload: str) -> Any | None:
        """读取 JSON 值。"""

        if not self._client:
            return None
        try:
            raw = self._client.get(self._key(prefix, payload))
            return json.loads(raw) if raw else None
        except Exception:  # noqa: BLE001
            return None

    def set_json(self, prefix: str, payload: str, value: Any, ttl_sec: int = 300) -> None:
        """写入带 TTL 的 JSON 值。"""

        if not self._client:
            return
        try:
            self._client.setex(
                self._key(prefix, payload), ttl_sec, json.dumps(value, ensure_ascii=False)
            )
        except Exception:  # noqa: BLE001
            pass
