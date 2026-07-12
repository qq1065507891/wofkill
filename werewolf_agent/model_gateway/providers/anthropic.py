# -*- coding: utf-8 -*-
"""
功能描述：Anthropic Messages API Provider 及响应解析器
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-09
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import json
import time
from typing import Any

from werewolf_agent.model_gateway.providers.base import _BaseHttpProvider
from werewolf_agent.model_gateway.providers.env import get_env
from werewolf_agent.model_gateway.router import GenerateResult, ModelConfig
from werewolf_agent.model_gateway.structured_output import (
    StructuredOutputMode,
    resolve_structured_output_mode,
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
        # P-N3 (post-review-v2): the legacy text-fallback path injected
        # a literal ``{"`` assistant message to "prime" the model into
        # starting its reply with ``{``. The mechanism was brittle:
        # the model would often return leading whitespace, a BOM, or
        # markdown code fences, which then collided with the
        # ``text[0] != "{"`` recovery below and produced invalid JSON.
        # We drop the priming entirely; the system prompt already
        # instructs the model to emit JSON, and the downstream
        # ``repair_json_text`` + ``json.loads`` chain handles
        # whitespace / BOM / fence cleanup.

        payload: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "messages": messages,
        }
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens
        if system_prompt:
            payload["system"] = system_prompt
        if tools and mode == StructuredOutputMode.NATIVE_TOOL:
            payload["tools"] = tools
        if tool_choice and mode == StructuredOutputMode.NATIVE_TOOL:
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
        # P-N3 (post-review-v2): the legacy code re-attached a ``{"``
        # prefix to the response when ``text[0] != "{"``. That
        # corrupted any response with leading whitespace, BOM, or
        # markdown fences. We return the text verbatim and let the
        # consumer parse it with ``repair_json_text`` + ``json.loads``,
        # which already strips leading whitespace and handles BOM.
        usage = data.get("usage", {})
        reasoning_tokens = int(
            (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        )
        has_thinking = any(
            item.get("type") == "thinking" for item in _anthropic_content(data)
        )
        return GenerateResult(
            text=text,
            provider=self.name,
            model=config.model,
            tool_call_required=forcing_tool,
            tool_call_received=tool_call_received,
            tool_call_name=(
                _anthropic_tool_name(data)
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
            reasoning_status=("confirmed" if has_thinking or reasoning_tokens else "not_requested"),
            reasoning_tokens=reasoning_tokens,
            usage=self._usage(
                model=config.model,
                latency_ms=latency_ms,
                prompt_tokens=int(usage.get("input_tokens", 0) or 0),
                completion_tokens=int(usage.get("output_tokens", 0) or 0),
            ),
        )


# -- Anthropic response parsers --


def _anthropic_content(data: dict[str, Any]) -> list[dict[str, Any]]:
    content = data.get("content") or []
    if not isinstance(content, list):
        if content is not None:
            import logging
            logging.getLogger(__name__).warning(
                "Unexpected Anthropic content type=%s raw=%s",
                type(content).__name__,
                repr(content)[:200],
            )
        return []
    return content


def _extract_anthropic_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in _anthropic_content(data):
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        elif item.get("type") == "tool_use":
            parts.append(json.dumps(item.get("input", {}), ensure_ascii=False))
    return "\n".join(part for part in parts if part)


def _has_anthropic_tool_use(data: dict[str, Any]) -> bool:
    return any(item.get("type") == "tool_use" for item in _anthropic_content(data))


def _anthropic_tool_name(data: dict[str, Any]) -> str:
    for item in _anthropic_content(data):
        if item.get("type") == "tool_use":
            return str(item.get("name", ""))
    return ""
