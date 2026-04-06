"""LLM 客户端封装模块。

统一屏蔽两种运行形态：
- 在线模式：调用 OpenAI 兼容接口；
- 离线模式：未配置 API Key 时返回可审计的 mock 响应，保证本地链路可跑通。
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from core.config.settings import Settings, get_settings
from core.observability.metrics import TOKENS_USED


class LLMClient:
    """面向项目内部的最小 LLM 适配层。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None
        if self._settings.openai_api_key:
            # 只有显式配置了 API Key 才启用在线调用，避免本地教学环境误打外网请求。
            self._client = AsyncOpenAI(
                api_key=self._settings.openai_api_key,
                base_url=self._settings.openai_base_url,
            )

    @property
    def enabled(self) -> bool:
        """是否启用真实 LLM。"""

        return self._client is not None

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        """执行一次非流式对话补全。"""

        if not self._client:
            return self._mock_response(messages)
        model = model or self._settings.llm_model_name
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        meta: dict[str, Any] = {}
        if usage:
            # 把 token 使用量记到 Prometheus，便于后续做成本估算和容量观察。
            TOKENS_USED.labels(model=model, kind="prompt").inc(usage.prompt_tokens or 0)
            TOKENS_USED.labels(model=model, kind="completion").inc(usage.completion_tokens or 0)
            meta = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            }
        return text, meta

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """执行流式补全，逐 token 产出内容。"""

        if not self._client:
            text, _ = self._mock_response(messages)
            yield text
            return
        model = model or self._settings.llm_model_name
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    def _mock_response(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        """离线模式下返回可审计占位文本。

        这样做的意义不是得到正确答案，而是保证：
        - API 链路可联调；
        - 前端能看到完整响应结构；
        - 测试环境不依赖真实外部模型服务。
        """

        last = messages[-1]["content"] if messages else ""
        return (
            "（离线模式）未配置 OPENAI_API_KEY。请根据上下文人工核对文档。"
            f" 最后提示摘要: {last[:200]}...",
            {"mock": True},
        )
