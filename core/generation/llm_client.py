"""LLM 客户端封装模块。

统一屏蔽两种运行形态：
- 在线模式：调用 OpenAI 兼容接口；
- 离线模式：未配置 API Key 时返回可审计的 mock 响应，保证本地链路可跑通。
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Literal

from openai import AsyncOpenAI

from core.config.settings import Settings, get_settings
from core.observability.metrics import TOKENS_USED


class LLMClient:
    """面向项目内部的最小 LLM 适配层。

    这个封装刻意保持很薄，只做三件事：
    1. 在线 / 离线两种运行形态切换
    2. 按任务选择模型
    3. 统一 token 指标采集

    它刻意不负责：
    - prompt 模板设计
    - 输出解析
    - 安全出域判断

    这样职责边界会更清楚：
    `LLMClient` 只关心“怎么调模型”，而不关心“为什么调、调完怎么解释”。
    """

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

    def _resolve_model(
        self,
        *,
        model: str | None = None,
        task: Literal["query_understanding", "query_planning", "answer_generation"] | None = None,
    ) -> str:
        """根据任务类型选择模型，允许显式覆盖。

        当前项目已经开始做任务级模型分层：
        - query understanding
        - query planning
        - answer generation

        这样可以避免所有任务都共用同一个大模型，方便后续做：
        - 成本控制
        - 时延优化
        - 任务级模型切换
        """

        if model:
            return model
        if task == "query_understanding":
            return self._settings.query_understanding_model_name
        if task == "query_planning":
            return self._settings.query_planning_model_name
        if task == "answer_generation":
            return self._settings.answer_generation_model_name
        return self._settings.llm_model_name

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        task: Literal["query_understanding", "query_planning", "answer_generation"] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        """执行一次非流式对话补全。

        非流式更适合：
        - query understanding
        - query planning
        - 完整生成后再统一解析结果的场景
        """

        if not self._client:
            return self._mock_response(messages)
        model = self._resolve_model(model=model, task=task)
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
        task: Literal["query_understanding", "query_planning", "answer_generation"] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """执行流式补全，逐 token 产出内容。

        这里故意不对外暴露 OpenAI 原始 stream 对象，
        而是统一收敛成“只产出文本片段”的接口，方便上层 API 直接转 NDJSON。

        这样 API 层不需要感知底层 SDK 的 chunk 结构，
        只要把字符串 token 往前端持续转发即可。
        """

        if not self._client:
            text, _ = self._mock_response(messages)
            yield text
            return
        model = self._resolve_model(model=model, task=task)
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

        所以这里返回的是“可说明当前处于离线模式”的占位内容，
        而不是伪装成真实正确答案。
        """

        last = messages[-1]["content"] if messages else ""
        return (
            "（离线模式）未配置 OPENAI_API_KEY。请根据上下文人工核对文档。"
            f" 最后提示摘要: {last[:200]}...",
            {"mock": True},
        )
