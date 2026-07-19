# -*- coding: utf-8 -*-
"""
功能描述：MiniMax Anthropic 兼容 Provider，注意该厂商不完全支持 tool_choice

作者：Mike
创建日期：2025-01-15
修改日期：2026-07-15

2026-07-15 新增：``config.base_url`` 覆盖 provider 实例默认 URL；``config.extra_body``
合并进 payload。用于与 OpenAI 客户端版本的 native MiniMax 共存。

使用示例：内部模块，无对外接口
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
        final_prompt_observer: FinalPromptObserver | None = None,
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
        # P-N3 (post-review-v2): removed ``{"`` priming and post-hoc
        # prefix injection. The legacy text-fallback path was brittle
        # against leading whitespace, BOM, or markdown fences. The
        # downstream ``repair_json_text`` + ``json.loads`` chain
        # handles all those cases. See anthropic.py for the
        # matching change and rationale.

        payload: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "messages": messages,
        }
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens
        if config.reasoning_requested:
            budget = 1024 if config.reasoning_level in {"medium", "high"} else 512
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            payload["max_tokens"] = max(int(config.max_tokens or 0), budget + 1024)
            payload["temperature"] = 1
        if system_prompt:
            payload["system"] = system_prompt
        if tools and mode == StructuredOutputMode.NATIVE_TOOL:
            payload["tools"] = tools
        # 2026-07-15: per-profile extra_body 合并。
        if config.extra_body:
            for key, value in config.extra_body.items():
                payload.setdefault(key, value)

        if final_prompt_observer is not None:
            system_content = str(payload.get("system") or "")
            notify_final_prompt_observer(final_prompt_observer, FinalPromptAssembly(
                system_bytes=system_content.encode("utf-8"),
                provider_payload_bytes=canonical_provider_payload(payload),
                final_system_location="system",
                final_system_message_index=None,
                provider=self.name,
                model=config.model,
            ))

        start = time.monotonic()
        response = self._http_client.post(
            f"{config.base_url or self._base_url}/v1/messages",
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
        # P-N3 (post-review-v2): removed ``{" + text`` re-attachment.
        # Return text verbatim; downstream parser handles it.
        usage = data.get("usage", {})
        reasoning_tokens = int(
            (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        )
        has_thinking = any(
            item.get("type") == "thinking"
            for item in (data.get("content") or [])
            if isinstance(item, dict)
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
            reasoning_status=(
                "confirmed"
                if has_thinking or reasoning_tokens
                else "requested_unconfirmed" if config.reasoning_requested
                else "not_requested"
            ),
            reasoning_tokens=reasoning_tokens,
            usage=self._usage(
                model=config.model,
                latency_ms=latency_ms,
                prompt_tokens=int(usage.get("input_tokens", 0) or 0),
                completion_tokens=int(usage.get("output_tokens", 0) or 0),
            ),
        )
