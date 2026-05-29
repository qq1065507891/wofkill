"""MiniMax Anthropic-compatible provider.

MiniMax exposes an Anthropic-compatible API but does NOT reliably
support tool_choice. When tool_choice is specified, MiniMax may
return plain text instead of a tool_use block.
"""

from __future__ import annotations

import time
from typing import Any

from werewolf_agent.model_gateway.providers.anthropic import (
    _anthropic_tool_name,
    _extract_anthropic_text,
    _has_anthropic_tool_use,
)
from werewolf_agent.model_gateway.providers.base import _BaseHttpProvider
from werewolf_agent.model_gateway.providers.env import get_env
from werewolf_agent.model_gateway.router import GenerateResult, ModelConfig


class MiniMaxProvider(_BaseHttpProvider):
    """MiniMax Anthropic-compatible provider."""

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
