# -*- coding: utf-8 -*-
"""
模型网关的配置、生成结果、用量记录与逐尝试执行记录兼容入口。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-13

使用示例:
    >>> from werewolf_agent.model_gateway.usage_records import ModelConfig
    >>> ModelConfig(provider="mock", model="mock").provider
    'mock'
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from werewolf_agent.model_gateway.execution_records import AttemptExecutionRecord


@dataclass(frozen=True)
class ModelConfig:
    """单次模型调用解析后的配置。"""
    provider: str
    model: str
    temperature: float = 0.5
    max_tokens: int | None = None
    top_p: float = 0.9
    timeout: int = 30
    allow_text_tool_fallback: bool = False
    retry_count: int = 2
    structured_output_mode: str = "auto"
    structured_output_fallback_modes: tuple[str, ...] = ()
    reasoning_level: str = "none"
    reasoning_requested: bool = False


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
    attempts: tuple[AttemptExecutionRecord, ...] = ()


@dataclass
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
    attempts: tuple[AttemptExecutionRecord, ...] = ()


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
    ) -> GenerateResult:
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
