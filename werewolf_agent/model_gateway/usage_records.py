# -*- coding: utf-8 -*-
"""
模型网关的配置、生成结果、用量记录与逐尝试执行记录兼容入口。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-23

使用示例:
    >>> from werewolf_agent.model_gateway.usage_records import ModelConfig
    >>> ModelConfig(provider="mock", model="mock").provider
    'mock'
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from werewolf_agent.model_gateway.execution_records import AttemptExecutionRecord
from werewolf_agent.model_gateway.final_prompt_observer import (
    FinalPromptObserver,
    notify_final_prompt_observer,
)


@dataclass(frozen=True)
class ModelConfig:
    """单次模型调用解析后的配置。

    ``base_url`` 与 ``extra_body`` 是 2026-07-15 新增字段，用于在不新增
    provider_name 的前提下，让同一 ``provider: openai`` 客户端同时服务
    多个 endpoint（例如 ``api.minimaxi.com/v1`` 与
    ``ark.cn-beijing.volces.com``）。``base_url=None`` 时回退到 provider
    实例默认 URL；``extra_body`` 在 payload 末尾合并进 JSON 请求体，覆盖
    默认字段必须用 kwargs spread。

    ``extra_body`` 为 dict，不可哈希，故显式 ``__hash__ = None``。
    路由侧把 ModelConfig 作为值对象传递，不放入 set/dict key，所以无影响。
    """
    provider: str
    model: str
    temperature: float = 0.5
    max_tokens: int | None = None
    top_p: float = 0.9
    timeout: int = 300
    allow_text_tool_fallback: bool = False
    retry_count: int = 4
    structured_output_mode: str = "auto"
    structured_output_fallback_modes: tuple[str, ...] = ()
    reasoning_level: str = "none"
    reasoning_requested: bool = False
    reasoning_capability: str = "none"
    base_url: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class UsageRecord:
    """单次模型调用的用量记录。"""
    agent_id: str
    task_type: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    estimated_cost: float = 0.0
    fallback_reason: str | None = None
    success: bool = True
    structured_output_mode: str = ""
    request_id: str = ""
    primary_provider: str = ""
    primary_model: str = ""
    fallback_provider: str | None = None
    fallback_model: str | None = None
    retry_count: int = 0
    failure_category: str | None = None
    reasoning_level: str = "none"
    reasoning_status: str = "not_requested"
    reasoning_tokens: int = 0
    # 2026-07-21 R2: Anthropic prompt cache 命中统计.
    # cache_creation_input_tokens = 首次写入 prefix 的 token (走 1.25x 计费).
    # cache_read_input_tokens = 复用 cache 的 token (走 0.1x 计费).
    # 仅 Anthropic / MiniMax (anthropic-compatible) 这两类厂商会填这两个字段;
    # OpenAI / GLM 默认 0.
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    attempts: tuple[AttemptExecutionRecord, ...] = ()

    def __post_init__(self) -> None:
        """把旧构造字段一次性归一化为最终 attempt 的只读视图。"""
        if not self.attempts:
            return
        final = self.attempts[-1]
        primary = self.attempts[0]
        object.__setattr__(self, "request_id", final.opaque_request_id.value)
        object.__setattr__(self, "provider", final.provider)
        object.__setattr__(self, "model", final.model)
        object.__setattr__(self, "primary_provider", primary.provider)
        object.__setattr__(self, "primary_model", primary.model)
        fallback = next(
            (item for item in self.attempts if item.route_kind.value == "provider_fallback"),
            None,
        )
        object.__setattr__(self, "fallback_provider", fallback.provider if fallback else None)
        object.__setattr__(self, "fallback_model", fallback.model if fallback else None)
        object.__setattr__(self, "success", final.attempt_outcome.value == "attempt_success")
        object.__setattr__(
            self,
            "retry_count",
            sum(item.route_kind.value == "retry" for item in self.attempts),
        )
        failures = [
            item for item in self.attempts
            if item.root_cause.value != "none"
            and item.route_kind.value != "safe_fallback"
        ]
        decisive_failure = (
            failures[-1]
            if failures and final.attempt_outcome.value == "attempt_failure"
            else failures[0] if failures else None
        )
        cause = decisive_failure.root_cause.value if decisive_failure else None
        object.__setattr__(self, "fallback_reason", cause)
        object.__setattr__(self, "failure_category", cause)
        object.__setattr__(self, "reasoning_level", final.requested_reasoning_level.value)
        object.__setattr__(self, "reasoning_status", final.normalized_reasoning_status.value)
        object.__setattr__(self, "reasoning_tokens", final.reasoning_token_count)


@dataclass(frozen=True)
class GenerateResult:
    """模型生成调用的返回结果。"""
    text: str
    provider: str
    model: str
    usage: UsageRecord | None = None
    tool_call_required: bool = False
    tool_call_received: bool = False
    tool_call_name: str = ""
    text_fallback_used: bool = False
    structured_failure_reason: str | None = None
    allow_text_tool_fallback: bool = False
    structured_output_mode: str = ""
    http_status: int = 0
    raw_error: str | None = None
    reasoning_level: str = "none"
    reasoning_status: str = "not_requested"
    reasoning_tokens: int = 0
    thinking_text: str = ""  # 2026-07-21: reasoning 原文, 不进 text。Ark reasoning_content / MiniMax <think> 剥离内容统一走此字段。
    attempts: tuple[AttemptExecutionRecord, ...] = ()

    def __post_init__(self) -> None:
        """在兼容构造边界固定旧 reasoning 视图，避免实例化后漂移。"""
        if self.usage and self.usage.attempts:
            if self.attempts and self.attempts != self.usage.attempts:
                raise ValueError("usage and result must share the same evidence chain")
            if not self.attempts:
                object.__setattr__(self, "attempts", self.usage.attempts)
        elif self.usage and self.attempts:
            object.__setattr__(self, "usage", replace(self.usage, attempts=self.attempts))
        if not self.attempts:
            return
        final = self.attempts[-1]
        object.__setattr__(self, "provider", final.provider)
        object.__setattr__(self, "model", final.model)
        object.__setattr__(self, "reasoning_level", final.requested_reasoning_level.value)
        object.__setattr__(self, "reasoning_status", final.normalized_reasoning_status.value)
        object.__setattr__(self, "reasoning_tokens", final.reasoning_token_count)


class EmptyModelResponseError(RuntimeError):
    """Provider 成功返回但模型文本为空。"""


class LLMProvider(Protocol):
    """LLM provider 协议。"""

    @property
    def name(self) -> str: ...

    def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        final_prompt_observer: FinalPromptObserver | None = None,
    ) -> GenerateResult: ...


class MockProvider:
    """测试用 mock provider，返回确定性占位文本。"""

    def __init__(self, name: str = "mock") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        final_prompt_observer: FinalPromptObserver | None = None,
    ) -> GenerateResult:
        if final_prompt_observer is not None and system_prompt:
            from werewolf_agent.model_gateway.final_prompt_observer import FinalPromptAssembly
            notify_final_prompt_observer(final_prompt_observer, FinalPromptAssembly(
                system_bytes=system_prompt.encode("utf-8"),
                final_system_location="messages",
                final_system_message_index=0,
                provider=self.name,
                model=config.model,
            ))
        start = time.monotonic()
        text = f"[{self._name}:{config.model}] mock response"
        latency = int((time.monotonic() - start) * 1000)
        return GenerateResult(
            text=text,
            provider=self._name,
            model=config.model,
            usage=UsageRecord(
                agent_id="",
                task_type="",
                provider=self._name,
                model=config.model,
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(text.split()),
                latency_ms=latency,
            ),
        )


__all__ = [
    "EmptyModelResponseError",
    "GenerateResult",
    "LLMProvider",
    "MockProvider",
    "ModelConfig",
    "UsageRecord",
]
