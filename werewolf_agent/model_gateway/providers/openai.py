# -*- coding: utf-8 -*-
"""
功能描述：OpenAI 兼容 Chat Completions Provider 及 GLM 共享生成辅助函数
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import time
from typing import Any

from werewolf_agent.model_gateway.providers.base import _BaseHttpProvider
from werewolf_agent.model_gateway.providers.env import get_env
from werewolf_agent.model_gateway.router import GenerateResult, ModelConfig
from werewolf_agent.model_gateway.structured_output import (
    StructuredOutputMode,
    resolve_structured_output_mode,
)


class OpenAIProvider(_BaseHttpProvider):
    """OpenAI-compatible Chat Completions provider."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key or get_env("OPENAI_API_KEY"),
            base_url=base_url or get_env("OPENAI_BASE_URL", "https://api.openai.com"),
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return "openai"

    def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> GenerateResult:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return _generate_openai_compatible(
            provider=self,
            base_url=self._base_url,
            api_key=self._api_key,
            http_client=self._http_client,
            messages=messages,
            config=config,
            tools=tools,
            tool_choice=tool_choice,
        )


# -- OpenAI-compatible generation (shared with GLM) --


def _generate_openai_compatible(
    *,
    provider: _BaseHttpProvider,
    base_url: str,
    api_key: str,
    http_client: Any,
    messages: list[dict[str, str]],
    config: ModelConfig,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
) -> GenerateResult:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
    }
    mode = resolve_structured_output_mode(
        provider=config.provider,
        configured_mode=config.structured_output_mode,
        allow_text_tool_fallback=config.allow_text_tool_fallback,
    )
    forcing_tool = bool(
        mode == StructuredOutputMode.NATIVE_TOOL
        and tool_choice
        and tool_choice.get("name")
    )
    if mode == StructuredOutputMode.JSON_SCHEMA and tools:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "player_action",
                "schema": _sanitize_for_json_schema(
                    tools[0].get("input_schema", {})
                ),
            },
        }
    elif mode == StructuredOutputMode.JSON_OBJECT:
        payload["response_format"] = {"type": "json_object"}
    elif mode == StructuredOutputMode.NATIVE_TOOL and tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            for tool in tools
        ]
        if forcing_tool:
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
    start = time.monotonic()
    response = http_client.post(
        _openai_chat_completions_url(base_url),
        headers={
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        json=payload,
        timeout=config.timeout,
    )
    response.raise_for_status()
    latency_ms = int((time.monotonic() - start) * 1000)
    data = response.json()
    message = data.get("choices", [{}])[0].get("message", {})
    tool_call_received = bool(message.get("tool_calls"))
    text = message.get("content", "") or _extract_openai_tool_text(message)
    usage = data.get("usage", {})
    return GenerateResult(
        text=text,
        provider=provider.name,
        model=config.model,
        tool_call_required=forcing_tool,
        tool_call_received=tool_call_received,
        tool_call_name=(
            _openai_tool_name(message)
            or ((tool_choice or {}).get("name", "") if forcing_tool else "")
        ),
        text_fallback_used=(
            bool(tools and forcing_tool and not tool_call_received and text)
            or (mode != StructuredOutputMode.NATIVE_TOOL and bool(text))
        ),
        structured_failure_reason=(
            "missing_tool_call"
            if tools and forcing_tool and not tool_call_received
            else None
        ),
        structured_output_mode=mode.value,
        usage=provider._usage(
            model=config.model,
            latency_ms=latency_ms,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        ),
    )


def _openai_chat_completions_url(base_url: str) -> str:
    """Build a chat completions URL for OpenAI and OpenAI-compatible gateways.

    R3-MG-10: the legacy helper accepted any URL whose last path segment
    contained the substring ``/v`` (e.g. ``/v1beta``, ``/v2``, ``/v1.5``)
    and appended ``/chat/completions`` directly. That produced 404s for
    real OpenAI-compatible gateways that use non-/v1 API versions
    (e.g. ``/v1beta/chat/completions``).

    P-N2 (post-review-v2): the post-R3-MG-10 "fall back to /v1" fix
    was itself too aggressive — it stripped declared version segments
    like ``/v1beta``, ``/v2beta``, ``/v4beta``, ``/v5alpha`` and
    re-routed to ``/v1/chat/completions``, which 404s on gateways that
    use the versioned path. Real OpenAI-compatible gateways
    (Zhipu / Baidu / Anthropic-compatible) treat ``vN`` / ``vNbeta`` /
    ``vNalpha`` as legitimate, stable API versions. The new rule
    preserves ANY versioned segment the caller declared.

    Rules:
    - last segment is ``vN`` (integer) or ``vN<letters>`` (e.g.
      ``v1``, ``v2``, ``v1beta``, ``v2beta``, ``v4beta``, ``v5alpha``):
      append ``/chat/completions`` directly.
    - last segment is something else (e.g. ``/coding``, ``/api``):
      preserve the path and append ``/chat/completions`` — preserves
      real-world OpenAI-compatible gateways like
      ``https://qianfan.baidubce.com/v2/coding`` whose /vN lives in
      the middle of the path, not at the end.
    - last segment is the bare host (no path at all) or default
      ``https://api.openai.com``: default to /v1/chat/completions.
    - URL already ends in ``/chat/completions``: return as-is.
    """
    import re
    normalized = base_url.rstrip("/")
    # Pass through if the URL is already a chat-completions URL.
    if normalized.endswith("/chat/completions"):
        return normalized
    last_segment = normalized.rsplit("/", 1)[-1] if normalized else ""
    # P-N2: preserve ANY versioned suffix (v1, v2, v1beta, v4beta,
    # v5alpha, ...). The R3-MG-10 "fall back to /v1" branch was
    # wrong — it stripped the caller's declared version.
    if re.fullmatch(r"v\d+\w*", last_segment):
        return f"{normalized}/chat/completions"
    # Non-versioned last segment with a path (e.g. /v2/coding) — keep
    # the path the caller declared and append /chat/completions.
    if normalized and normalized != "https://api.openai.com":
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _sanitize_for_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt an Anthropic-format input_schema for OpenAI json_schema."""
    import copy
    s = copy.deepcopy(schema)

    def _walk(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        t = node.get("type")
        if isinstance(t, list) and "null" in t:
            non_null = [x for x in t if x != "null"]
            value_schema = {
                key: value
                for key, value in node.items()
                if key not in {"type", "description", "title"}
            }
            value_schema["type"] = (
                non_null[0] if len(non_null) == 1 else non_null
            )
            if "enum" in value_schema:
                value_schema["enum"] = [
                    value for value in value_schema["enum"]
                    if value is not None
                ]
            metadata = {
                key: node[key]
                for key in ("description", "title")
                if key in node
            }
            node.clear()
            node.update(metadata)
            node["anyOf"] = [_walk(value_schema), {"type": "null"}]
            return node
        for _k, v in node.items():
            if isinstance(v, dict):
                _walk(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _walk(item)
        return node

    return _walk(s)


def _extract_openai_tool_text(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    for call in calls:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if arguments:
            return str(arguments)
    return ""


def _openai_tool_name(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    for call in calls:
        function = call.get("function") or {}
        name = function.get("name")
        if name:
            return str(name)
    return ""
