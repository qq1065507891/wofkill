"""HTTP LLM providers for ModelRouter.

API keys are read from environment variables or an untracked .env file.
The router still depends only on the LLMProvider protocol; provider SDKs are
not required for the V1 runtime.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from werewolf_agent.model_gateway.router import GenerateResult, ModelConfig, UsageRecord


PROVIDER_DOTENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "GLM_API_KEY",
    "GLM_BASE_URL",
    "SILICONFLOW_API_KEY",
    "SILICONFLOW_BASE_URL",
    "MINIMAX_API_KEY",
    "MINIMAX_BASE_URL",
}


class ProviderConfigError(RuntimeError):
    """Raised when a provider cannot be configured safely."""


class _BaseHttpProvider:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        http_client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ProviderConfigError(f"{self.name} API key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client or httpx.Client()
        self._owns_http = http_client is None

    def close(self) -> None:
        if self._owns_http:
            self._http_client.close()

    def __enter__(self) -> "_BaseHttpProvider":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def name(self) -> str:
        raise NotImplementedError

    def _usage(
        self,
        *,
        model: str,
        latency_ms: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> UsageRecord:
        return UsageRecord(
            agent_id="",
            task_type="",
            provider=self.name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )


class AnthropicProvider(_BaseHttpProvider):
    """Anthropic Messages API provider."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key or get_env("ANTHROPIC_API_KEY"),
            base_url=base_url or get_env("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return "anthropic"

    def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> GenerateResult:
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        # Prefill assistant with "{" to force JSON output from the first token.
        # Only for text-fallback models without forced tool_choice — mutually
        # exclusive with tool_use (the API rejects prefill + tool_choice).
        forcing_tool = bool(tool_choice and tool_choice.get("name"))
        if config.allow_text_tool_fallback and not forcing_tool:
            messages.append({"role": "assistant", "content": "{"})

        payload: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "messages": messages,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools and not (config.allow_text_tool_fallback and not forcing_tool):
            payload["tools"] = tools
        if tool_choice and not (config.allow_text_tool_fallback and not forcing_tool):
            payload["tool_choice"] = tool_choice

        start = time.monotonic()
        response = self._http_client.post(
            f"{self._base_url}/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=config.timeout,
        )
        response.raise_for_status()
        latency_ms = int((time.monotonic() - start) * 1000)
        data = response.json()
        tool_call_received = _has_anthropic_tool_use(data)
        text = _extract_anthropic_text(data)
        # When prefilled with "{", prepend it back to the response text
        # so the JSON is complete. The API strips the prefill from output.
        if config.allow_text_tool_fallback and not forcing_tool and text:
            text = "{" + text if text[0] != "{" else text
        usage = data.get("usage", {})
        return GenerateResult(
            text=text,
            provider=self.name,
            model=config.model,
            tool_call_required=bool(tool_choice),
            tool_call_received=tool_call_received,
            tool_call_name=_anthropic_tool_name(data) or (tool_choice or {}).get("name", ""),
            text_fallback_used=bool(tools and tool_choice and not tool_call_received and text)
                or (config.allow_text_tool_fallback and not forcing_tool and bool(text)),
            structured_failure_reason=(
                "missing_tool_call" if tools and tool_choice and not tool_call_received else None
            ),
            usage=self._usage(
                model=config.model,
                latency_ms=latency_ms,
                prompt_tokens=int(usage.get("input_tokens", 0) or 0),
                completion_tokens=int(usage.get("output_tokens", 0) or 0),
            ),
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


class GLMProvider(_BaseHttpProvider):
    """ZhipuAI/GLM OpenAI-compatible provider."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key or get_env("GLM_API_KEY"),
            base_url=base_url or get_env("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return "glm"

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


class MiniMaxProvider(_BaseHttpProvider):
    """MiniMax Anthropic-compatible provider.

    MiniMax exposes an Anthropic-compatible API but does NOT reliably
    support tool_choice.  When tool_choice is specified, MiniMax may
    return plain text instead of a tool_use block.  This provider:
      - Sends tools in the request (so the model knows the schema)
      - Strips tool_choice from the payload (MiniMax ignores it anyway)
      - Extracts structured JSON from either tool_use or plain text
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key or get_env("MINIMAX_API_KEY")
                or get_env("ANTHROPIC_API_KEY"),
            base_url=base_url or get_env("MINIMAX_BASE_URL")
                or get_env("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic"),
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return "minimax"

    def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> GenerateResult:
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        forcing_tool = bool(tool_choice and tool_choice.get("name"))
        if config.allow_text_tool_fallback and not forcing_tool:
            messages.append({"role": "assistant", "content": "{"})

        payload: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "messages": messages,
        }
        if system_prompt:
            payload["system"] = system_prompt
        # Send tools so model knows the schema, but DO NOT send tool_choice
        # — MiniMax does not reliably enforce it.  Skip tools entirely when
        # using prefill (assistant message already forces JSON shape).
        if tools and not (config.allow_text_tool_fallback and not forcing_tool):
            payload["tools"] = tools

        start = time.monotonic()
        response = self._http_client.post(
            f"{self._base_url}/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=config.timeout,
        )
        response.raise_for_status()
        latency_ms = int((time.monotonic() - start) * 1000)
        data = response.json()

        # Extract text: prefer tool_use input, fall back to plain text
        tool_call_received = _has_anthropic_tool_use(data)
        text = _extract_anthropic_text(data)
        if config.allow_text_tool_fallback and not forcing_tool and text:
            text = "{" + text if text[0] != "{" else text
        usage = data.get("usage", {})
        return GenerateResult(
            text=text,
            provider=self.name,
            model=config.model,
            tool_call_required=bool(tool_choice),
            tool_call_received=tool_call_received,
            tool_call_name=_anthropic_tool_name(data) or (tool_choice or {}).get("name", ""),
            text_fallback_used=bool(tools and tool_choice and not tool_call_received and text)
                or (config.allow_text_tool_fallback and not forcing_tool and bool(text)),
            structured_failure_reason=(
                "missing_tool_call" if tools and tool_choice and not tool_call_received else None
            ),
            usage=self._usage(
                model=config.model,
                latency_ms=latency_ms,
                prompt_tokens=int(usage.get("input_tokens", 0) or 0),
                completion_tokens=int(usage.get("output_tokens", 0) or 0),
            ),
        )


def create_provider_from_env(provider_name: str):
    """Create a known provider only when its API key is present."""
    load_local_dotenv()
    normalized = provider_name.lower()
    if normalized == "anthropic" and get_env("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    if normalized == "openai" and get_env("OPENAI_API_KEY"):
        return OpenAIProvider()
    if normalized == "glm" and get_env("GLM_API_KEY"):
        return GLMProvider()
    if normalized == "minimax" and (
        get_env("MINIMAX_API_KEY") or get_env("ANTHROPIC_API_KEY")
    ):
        return MiniMaxProvider()
    return None


_ENV_OVERRIDES: dict[str, str] = {}


def get_env(key: str, default: str = "") -> str:
    """Read env var, preferring .env overrides loaded by load_local_dotenv."""
    return _ENV_OVERRIDES.get(key) or os.getenv(key, default)


def load_local_dotenv(
    path: str | Path = ".env",
    *,
    keys: set[str] | None = None,
) -> None:
    """Load provider KEY=VALUE lines without requiring python-dotenv at runtime."""
    env_path = Path(path)
    if not env_path.exists():
        return
    allowed_keys = keys or PROVIDER_DOTENV_KEYS
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed_keys:
            continue
        value = value.strip().strip('"').strip("'")
        if key and value:
            _ENV_OVERRIDES[key] = value


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
        # No tools available — fall back to basic json_object constraint
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
    """Adapt an Anthropic-format input_schema for OpenAI json_schema.

    Fixes: ``"type": ["string","null"]`` → string-only,
    removes ``None`` from enum lists.
    """
    import copy
    s = copy.deepcopy(schema)

    def _walk(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        t = node.get("type")
        # Replace ["string","null"] with "string" + nullable marker.
        # OpenAI json_schema does not accept type arrays in many backends.
        if isinstance(t, list) and "null" in t:
            non_null = [x for x in t if x != "null"]
            node["type"] = non_null[0] if len(non_null) == 1 else non_null
        # Strip None from enum values — json_schema backends reject it
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


def _extract_anthropic_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in data.get("content", []):
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        elif item.get("type") == "tool_use":
            parts.append(json.dumps(item.get("input", {}), ensure_ascii=False))
    return "\n".join(part for part in parts if part)


def _has_anthropic_tool_use(data: dict[str, Any]) -> bool:
    return any(item.get("type") == "tool_use" for item in data.get("content", []))


def _anthropic_tool_name(data: dict[str, Any]) -> str:
    for item in data.get("content", []):
        if item.get("type") == "tool_use":
            return str(item.get("name", ""))
    return ""


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
