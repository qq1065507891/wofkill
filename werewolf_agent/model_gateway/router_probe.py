# -*- coding: utf-8 -*-
"""
检测模型 provider 是否真正支持结构化工具调用。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.model_gateway.router_probe import probe_tool_call_support
    >>> callable(probe_tool_call_support)
    True
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

from werewolf_agent.model_gateway.provider_call import (
    _call_provider_generate,
    _normalize_tool_metadata,
)
from werewolf_agent.model_gateway.retry_policy import _format_exception
from werewolf_agent.model_gateway.structured_output import StructuredOutputMode
from werewolf_agent.model_gateway.usage_records import LLMProvider, ModelConfig


class RouterProbeContext(Protocol):
    """probe helper 需要的最小 router 接口。"""

    _providers: dict[str, LLMProvider]

    def resolve_config(
        self,
        agent_id: str,
        task_type: str,
    ) -> tuple[ModelConfig, str | None]: ...


def probe_tool_call_support(
    router: RouterProbeContext,
    agent_id: str,
    task_type: str,
) -> dict[str, Any]:
    """探测已解析 provider 是否返回真实 tool call。"""
    config, _fallback_provider = router.resolve_config(agent_id, task_type)
    config = replace(
        config,
        structured_output_mode=StructuredOutputMode.NATIVE_TOOL.value,
    )
    provider = router._providers.get(config.provider)
    if provider is None:
        raise RuntimeError(
            f"Provider '{config.provider}' not found. "
            f"Available: {list(router._providers.keys())}"
        )

    tool = {
        "name": "submit_player_action",
        "description": "Probe structured action tool-call support.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action_type": {"type": "string", "enum": ["no_action"]},
                "target_id": {"enum": [None]},
                "speech": {"type": "string"},
                "reason": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "action_type",
                "target_id",
                "speech",
                "reason",
                "confidence",
            ],
        },
    }
    tool_choice = {"type": "tool", "name": "submit_player_action"}
    try:
        result = _call_provider_generate(
            provider,
            "Call submit_player_action with no_action probe arguments.",
            config,
            "You are checking tool-call support. Use the tool.",
            tools=[tool],
            tool_choice=tool_choice,
        )
        _normalize_tool_metadata(result, tool_choice)
    except Exception as exc:
        return {
            "supported": False,
            "provider": config.provider,
            "model": config.model,
            "failure_reason": _format_exception(exc),
            "tool_call_received": False,
            "text_fallback_used": False,
        }
    supported = bool(result.tool_call_received) and not result.structured_failure_reason
    return {
        "supported": supported,
        "provider": result.provider,
        "model": result.model,
        "failure_reason": result.structured_failure_reason,
        "tool_call_received": result.tool_call_received,
        "tool_call_name": result.tool_call_name,
        "text_fallback_used": result.text_fallback_used,
    }


__all__ = [
    "probe_tool_call_support",
]
