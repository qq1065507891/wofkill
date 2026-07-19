# -*- coding: utf-8 -*-
"""
功能描述：OpenAI 兼容 Chat Completions Provider 及 GLM 共享生成辅助函数

作者：Mike
创建日期：2025-01-15
修改日期：2026-07-19

支持 ``config.base_url`` 覆盖 provider 实例默认 URL（2026-07-15），
用于同一 OpenAI 客户端服务多个 endpoint（``api.minimaxi.com/v1`` 与
``ark.cn-beijing.volces.com``）。``config.extra_body`` 在 payload 末尾合并，
可携带 ``reasoning_split`` 等厂商私有开关。

native MiniMax endpoint 按 dedicated、vendor、scoped Anthropic key 解析，
缺少合法 key 时 fail closed，避免回退到 Ark 的 ``OPENAI_API_KEY``。

使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from werewolf_agent.model_gateway.providers.base import (
    ProviderConfigError,
    _BaseHttpProvider,
)
from werewolf_agent.model_gateway.final_prompt_observer import (
    FinalPromptAssembly,
    FinalPromptObserver,
    canonical_provider_payload,
    notify_final_prompt_observer,
)
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
        final_prompt_observer: FinalPromptObserver | None = None,
    ) -> GenerateResult:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        base_url = config.base_url or self._base_url
        return _generate_openai_compatible(
            provider=self,
            base_url=base_url,
            api_key=_resolve_api_key_for_base_url(base_url, self._api_key),
            http_client=self._http_client,
            messages=messages,
            config=config,
            tools=tools,
            tool_choice=tool_choice,
            final_prompt_observer=final_prompt_observer,
        )


# -- OpenAI-compatible generation (shared with GLM) --


_MINIMAX_NATIVE_HOST = "api.minimaxi.com"


def _resolve_api_key_for_base_url(base_url: str, default_key: str) -> str:
    """按最终请求 URL 解析 OpenAI-compatible endpoint 的鉴权键。"""
    parsed_url = urlparse(base_url)
    if parsed_url.hostname != _MINIMAX_NATIVE_HOST:
        return default_key
    if parsed_url.scheme != "https":
        raise ProviderConfigError("native MiniMax endpoint requires HTTPS")

    # native 专用键优先，其次使用 MiniMax 厂商键。
    for key_name in ("MINIMAX_NATIVE_API_KEY", "MINIMAX_API_KEY"):
        api_key = get_env(key_name)
        if api_key:
            return api_key

    # 仅当 Anthropic endpoint 同样精确指向 MiniMax 时，才能复用其鉴权键。
    anthropic_base_url = get_env("ANTHROPIC_BASE_URL")
    if urlparse(anthropic_base_url).hostname == _MINIMAX_NATIVE_HOST:
        anthropic_key = get_env("ANTHROPIC_API_KEY")
        if anthropic_key:
            return anthropic_key
    raise ProviderConfigError(
        "native MiniMax API key is required: set MINIMAX_NATIVE_API_KEY or "
        "MINIMAX_API_KEY, or set ANTHROPIC_API_KEY with ANTHROPIC_BASE_URL "
        "scoped to api.minimaxi.com"
    )


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
    final_prompt_observer: FinalPromptObserver | None = None,
) -> GenerateResult:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    if config.max_tokens is not None:
        payload[
            "max_completion_tokens" if config.reasoning_requested else "max_tokens"
        ] = config.max_tokens
    if config.reasoning_requested and config.reasoning_level != "none":
        payload["reasoning_effort"] = config.reasoning_level
    # 2026-07-15: per-profile extra_body 合并。必须在 structured_output /
    # tool_choice 之后追加，避免覆盖同名显式字段（payload[...] = 更靠右的赋值优先）。
    if config.extra_body:
        for key, value in config.extra_body.items():
            payload.setdefault(key, value)
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
    if final_prompt_observer is not None:
        system_index = next(
            (index for index, message in enumerate(payload["messages"])
             if message.get("role") == "system"),
            None,
        )
        system_content = (
            str(payload["messages"][system_index].get("content") or "")
            if system_index is not None else ""
        )
        notify_final_prompt_observer(final_prompt_observer, FinalPromptAssembly(
            system_bytes=system_content.encode("utf-8"),
            provider_payload_bytes=canonical_provider_payload(payload),
            final_system_location="messages",
            final_system_message_index=system_index,
            provider=provider.name,
            model=config.model,
        ))
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
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = int(details.get("reasoning_tokens", 0) or 0)
    message_reasoning = message.get("reasoning_content") or message.get("reasoning")
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
        reasoning_status=(
            "confirmed"
            if reasoning_tokens or message_reasoning
            else "requested_unconfirmed" if config.reasoning_requested
            else "not_requested"
        ),
        reasoning_tokens=reasoning_tokens,
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
