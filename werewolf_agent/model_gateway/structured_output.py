# -*- coding: utf-8 -*-
"""
功能描述：结构化输出协议选择与失败分类模块
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class StructuredOutputMode(str, Enum):
    NATIVE_TOOL = "native_tool"
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"
    TEXT_JSON = "text_json"


class StructuredFailureStage(str, Enum):
    PROVIDER = "provider"
    PROTOCOL = "protocol"
    SCHEMA = "schema"
    SEMANTIC = "semantic"


_PROTOCOL_FAILURES = {
    "missing_tool_call",
    "parse_error",
    "empty_response",
    "structured_output_unsupported",
}
_SCHEMA_FAILURES = {"schema_validation"}
_PROVIDER_FAILURES = {
    "model_generation_failed",
    "provider_error",
    "network_error",
    "timeout",
}


@dataclass(frozen=True)
class StructuredOutputPolicy:
    primary_mode: StructuredOutputMode
    fallback_modes: tuple[StructuredOutputMode, ...] = ()

    @property
    def modes(self) -> tuple[StructuredOutputMode, ...]:
        return (self.primary_mode, *self.fallback_modes)

    @classmethod
    def from_model_profile(
        cls,
        *,
        provider: str,
        model_profile: dict[str, Any],
    ) -> "StructuredOutputPolicy":
        configured = model_profile.get("structured_output") or {}
        if configured:
            primary = _coerce_mode(configured.get("mode"))
            fallbacks = tuple(
                _coerce_mode(mode)
                for mode in configured.get("fallback_modes", ())
            )
            return cls(primary, _dedupe_modes(primary, fallbacks))

        allow_text = bool(model_profile.get("allow_text_tool_fallback", False))
        primary = legacy_structured_output_mode(provider, allow_text)
        return cls(primary)

    @classmethod
    def from_config(cls, config: Any) -> "StructuredOutputPolicy":
        primary = resolve_structured_output_mode(
            provider=str(getattr(config, "provider", "")),
            configured_mode=str(
                getattr(config, "structured_output_mode", "auto")
            ),
            allow_text_tool_fallback=bool(
                getattr(config, "allow_text_tool_fallback", False)
            ),
        )
        fallbacks = tuple(
            _coerce_mode(mode)
            for mode in getattr(
                config,
                "structured_output_fallback_modes",
                (),
            )
        )
        return cls(primary, _dedupe_modes(primary, fallbacks))

    def next_mode(
        self,
        current_mode: StructuredOutputMode,
        failure_code: str | None,
    ) -> StructuredOutputMode:
        if classify_structured_failure(failure_code) not in {
            StructuredFailureStage.PROTOCOL,
            StructuredFailureStage.SCHEMA,
        }:
            return current_mode
        try:
            index = self.modes.index(current_mode)
        except ValueError:
            return self.primary_mode
        if index + 1 >= len(self.modes):
            return current_mode
        return self.modes[index + 1]


def legacy_structured_output_mode(
    provider: str,
    allow_text_tool_fallback: bool,
) -> StructuredOutputMode:
    if not allow_text_tool_fallback:
        return StructuredOutputMode.NATIVE_TOOL
    if provider in {"openai", "glm"}:
        return StructuredOutputMode.JSON_SCHEMA
    return StructuredOutputMode.TEXT_JSON


def resolve_structured_output_mode(
    *,
    provider: str,
    configured_mode: str,
    allow_text_tool_fallback: bool,
) -> StructuredOutputMode:
    if not configured_mode or configured_mode == "auto":
        return legacy_structured_output_mode(
            provider,
            allow_text_tool_fallback,
        )
    return _coerce_mode(configured_mode)


def classify_structured_failure(
    failure_code: str | None,
) -> StructuredFailureStage | None:
    if not failure_code:
        return None
    if failure_code in _PROVIDER_FAILURES:
        return StructuredFailureStage.PROVIDER
    if failure_code in _PROTOCOL_FAILURES:
        return StructuredFailureStage.PROTOCOL
    if failure_code in _SCHEMA_FAILURES:
        return StructuredFailureStage.SCHEMA
    return StructuredFailureStage.SEMANTIC


def _coerce_mode(value: Any) -> StructuredOutputMode:
    try:
        return StructuredOutputMode(str(value))
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in StructuredOutputMode)
        raise ValueError(
            f"Unknown structured output mode {value!r}; expected one of {valid}"
        ) from exc


def _dedupe_modes(
    primary: StructuredOutputMode,
    modes: tuple[StructuredOutputMode, ...],
) -> tuple[StructuredOutputMode, ...]:
    result: list[StructuredOutputMode] = []
    for mode in modes:
        if mode != primary and mode not in result:
            result.append(mode)
    return tuple(result)
