# -*- coding: utf-8 -*-
"""
功能描述：Anthropic Messages API Provider 及响应解析器

作者：Mike
创建日期：2025-01-15
修改日期：2026-07-21

2026-07-15 新增：``config.base_url`` 覆盖 provider 实例默认 URL；``config.extra_body``
合并进 payload（不覆盖已有字段）。用于同一 Anthropic 兼容客户端服务多个 endpoint。
2026-07-21 新增：anthropic prompt cache (cache_control: ephemeral) — system_prompt
非空时切到 list-of-text-blocks 形式, 给首 block 加 cache_control 标记, 跨轮
跨玩家复用 cache_read_input_tokens. 用量记录补充 cache_creation_input_tokens
和 cache_read_input_tokens 字段.

使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import json
import time
from typing import Any

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


# Anthropic prompt cache 标记. Claude 系列 ephemeral cache 的最小有效
# prefix 是 1024 tokens (claude-sonnet-4.5 / opus-4.8); 短 prefix 加 marker
# 也合法, 但 Anthropic 不会创建 cache, 不会额外收费. 我们对任意长度的
# system_prompt 都加 marker, 跨轮稳定够长时才命中 cache_read_input_tokens.
_ANTHROPIC_CACHE_CONTROL_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}


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
        if config.reasoning_requested:
            budget = 1024 if config.reasoning_level in {"medium", "high"} else 512
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            payload["max_tokens"] = max(int(config.max_tokens or 0), budget + 1024)
            payload["temperature"] = 1
        if system_prompt:
            payload["system"] = _wrap_system_prompt_for_cache(system_prompt)
        if tools and mode == StructuredOutputMode.NATIVE_TOOL:
            payload["tools"] = tools
        if tool_choice and mode == StructuredOutputMode.NATIVE_TOOL:
            payload["tool_choice"] = tool_choice
        # 2026-07-15: per-profile extra_body 合并，setdefault 不覆盖显式字段。
        if config.extra_body:
            for key, value in config.extra_body.items():
                payload.setdefault(key, value)

        if final_prompt_observer is not None:
            notify_final_prompt_observer(final_prompt_observer, FinalPromptAssembly(
                system_bytes=_system_bytes_for_observer(payload.get("system")),
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
        # 2026-07-21 R2: Anthropic prompt cache 统计字段.
        cache_creation_input_tokens = int(
            usage.get("cache_creation_input_tokens", 0) or 0
        )
        cache_read_input_tokens = int(
            usage.get("cache_read_input_tokens", 0) or 0
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
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            ),
        )


# 2026-07-21 R2: 把 system_prompt 包装成 Anthropic prompt cache 的 list 形式.
#
# 输入是 str. 输出 list-of-text-blocks 形式 (Anthropic 2026 协议) ::
#
#     [
#         {"type": "text", "text": "<system_prompt>",
#          "cache_control": {"type": "ephemeral"}},
#     ]
#
# 字节兼容 finalize_prompt_observer 的 contract marker 校验: marker 字符串
# 仍然以 Python repr 形式出现在 system_bytes 里, find() 仍命中.
def _wrap_system_prompt_for_cache(system_prompt: str) -> list[dict[str, Any]]:
    """Wrap a system_prompt string into Anthropic text-blocks with cache_control."""
    return [{
        "type": "text",
        "text": system_prompt,
        "cache_control": _ANTHROPIC_CACHE_CONTROL_EPHEMERAL,
    }]  # type: ignore[list-item]  # 单元素 list, 与 anthropic.py:list[dict] 形态兼容.


def _system_bytes_for_observer(system_value: str | list[dict[str, Any]] | None) -> bytes:
    """序列化 system 字段供 FinalPromptAssembly.system_bytes.

    兼容 str (旧用法) 和 list-of-text-blocks (R2 新形态).
    列表形式用 canonical JSON 保证后续调用者能字节化, 同时 contract marker
    仍然出现在 JSON 内的中文 UTF-8 序列里, find() 命中.
    """
    if system_value is None:
        return b""
    if isinstance(system_value, str):
        return system_value.encode("utf-8")
    # list 形式, 用 json.dumps 而非 str() (Python repr 会带单引号, 与 JSON 不同).
    return json.dumps(
        system_value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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
