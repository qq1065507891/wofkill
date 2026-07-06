# -*- coding: utf-8 -*-
"""
模型 provider 调用适配与结构化工具调用元数据归一化。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.model_gateway.provider_call import _normalize_tool_metadata
    >>> _normalize_tool_metadata(result, {"type": "tool", "name": "submit"})
"""

from __future__ import annotations

import inspect
from typing import Any

from werewolf_agent.model_gateway.usage_records import GenerateResult, LLMProvider, ModelConfig


def _call_provider_generate(
    provider: LLMProvider,
    prompt: str,
    config: ModelConfig,
    system_prompt: str | None,
    *,
    tools: list[dict[str, Any]] | None,
    tool_choice: dict[str, Any] | None,
) -> GenerateResult:
    signature = inspect.signature(provider.generate)
    if "tools" in signature.parameters:
        return provider.generate(
            prompt,
            config,
            system_prompt,
            tools=tools,
            tool_choice=tool_choice,
        )
    return provider.generate(prompt, config, system_prompt)


def _normalize_tool_metadata(
    result: GenerateResult,
    tool_choice: dict[str, Any] | str | None,
) -> None:
    if not tool_choice:
        return
    tc_type = tool_choice if isinstance(tool_choice, str) else tool_choice.get("type", "")
    if tc_type == "auto":
        return
    if result.allow_text_tool_fallback and result.text:
        result.tool_call_required = True
        if not result.tool_call_name:
            result.tool_call_name = str(tool_choice.get("name") or "") if isinstance(tool_choice, dict) else ""
        result.text_fallback_used = True
        return
    result.tool_call_required = True
    if not result.tool_call_name:
        result.tool_call_name = str(tool_choice.get("name") or "") if isinstance(tool_choice, dict) else ""
    if not result.tool_call_received:
        result.text_fallback_used = bool(result.text)
        if result.structured_failure_reason is None:
            result.structured_failure_reason = "missing_tool_call"


__all__ = [
    "_call_provider_generate",
    "_normalize_tool_metadata",
]
