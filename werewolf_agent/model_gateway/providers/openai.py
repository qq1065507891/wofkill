"""OpenAI-compatible Chat Completions provider and shared helpers."""

from __future__ import annotations

import time
from typing import Any

from werewolf_agent.model_gateway.providers.base import _BaseHttpProvider
from werewolf_agent.model_gateway.providers.env import get_env
from werewolf_agent.model_gateway.router import GenerateResult, ModelConfig


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
    forcing_tool = bool(tool_choice and tool_choice.get("name"))
    if tools and config.allow_text_tool_fallback and not forcing_tool:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "player_action",
                "schema": _sanitize_for_json_schema(
                    tools[0].get("input_schema", {})
                ),
            },
        }
    elif tools:
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
    elif config.allow_text_tool_fallback:
        payload["response_format"] = {"type": "json_object"}
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
        tool_call_required=bool(tool_choice),
        tool_call_received=tool_call_received,
        tool_call_name=_openai_tool_name(message) or (tool_choice or {}).get("name", ""),
        text_fallback_used=bool(tools and tool_choice and not tool_call_received and text)
            or (config.allow_text_tool_fallback and not forcing_tool and bool(text)),
        structured_failure_reason=(
            "missing_tool_call" if tools and tool_choice and not tool_call_received else None
        ),
        usage=provider._usage(
            model=config.model,
            latency_ms=latency_ms,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        ),
    )


def _openai_chat_completions_url(base_url: str) -> str:
    """Build a chat completions URL for OpenAI and OpenAI-compatible gateways."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1") or "/v" in normalized.rsplit("/", 1)[-1]:
        return f"{normalized}/chat/completions"
    if normalized != "https://api.openai.com" and "/" in normalized.removeprefix("https://").removeprefix("http://"):
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
            node["type"] = non_null[0] if len(non_null) == 1 else non_null
        if "enum" in node:
            node["enum"] = [v for v in node["enum"] if v is not None]
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
