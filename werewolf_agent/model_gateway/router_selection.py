# -*- coding: utf-8 -*-
"""
模型路由器配置选择与 fallback 模型解析。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-09

使用示例:
    >>> _resolve_config(model_profiles={}, llm_profiles={}, player_assignments={}, agent_id="p01", task_type="speech")[0].provider
    'mock'
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.model_gateway.structured_output import StructuredOutputPolicy
from werewolf_agent.model_gateway.usage_records import ModelConfig


def _resolve_config(
    *,
    model_profiles: dict[str, dict[str, Any]],
    llm_profiles: dict[str, dict[str, Any]],
    player_assignments: dict[str, str],
    agent_id: str,
    task_type: str,
) -> tuple[ModelConfig, str | None]:
    """按 agent 与任务解析主模型配置和 fallback provider。"""
    llm_profile_id = player_assignments.get(agent_id, "")
    llm_profile = llm_profiles.get(llm_profile_id, {})

    task_cfg = llm_profile.get("tasks", {}).get(task_type)
    default_cfg = llm_profile.get("default", {})
    source = task_cfg or default_cfg

    if not source:
        return ModelConfig(provider="mock", model="mock"), "mock"

    provider_name = source.get("provider", "mock")
    model_profile_id = source.get("model_profile", "")
    model_profile = model_profiles.get(model_profile_id, {})
    structured_policy = StructuredOutputPolicy.from_model_profile(
        provider=provider_name,
        model_profile=model_profile,
    )

    config = ModelConfig(
        provider=provider_name,
        model=model_profile.get("model", model_profile_id),
        temperature=model_profile.get("temperature", 0.5),
        max_tokens=model_profile.get("max_tokens"),
        top_p=model_profile.get("top_p", 0.9),
        timeout=model_profile.get("timeout", 30),
        allow_text_tool_fallback=bool(model_profile.get("allow_text_tool_fallback", False)),
        retry_count=int(model_profile.get("retry_count", 2)),
        structured_output_mode=structured_policy.primary_mode.value,
        structured_output_fallback_modes=tuple(
            mode.value for mode in structured_policy.fallback_modes
        ),
        reasoning_level=_reasoning_level(model_profile),
        reasoning_requested=bool(_reasoning_level(model_profile) != "none"),
    )

    fallback_cfg = llm_profile.get("fallback")
    fallback_provider = None
    if fallback_cfg:
        fallback_provider = fallback_cfg.get("provider")

    return config, fallback_provider


def _resolve_fallback_model(
    *,
    model_profiles: dict[str, dict[str, Any]],
    llm_profiles: dict[str, dict[str, Any]],
    llm_profile_id: str,
) -> ModelConfig | None:
    """解析 fallback model_profile，缺失引用时快速报错。"""
    from werewolf_agent.model_gateway.providers.base import ProviderConfigError

    llm_profile = llm_profiles.get(llm_profile_id, {})
    fallback_cfg = llm_profile.get("fallback", {})
    if not fallback_cfg:
        return None
    model_profile_id = fallback_cfg.get("model_profile", "")
    if not model_profile_id:
        raise ProviderConfigError(
            f"llm_profile {llm_profile_id!r}.fallback has no model_profile"
        )
    if model_profile_id not in model_profiles:
        raise ProviderConfigError(
            f"llm_profile {llm_profile_id!r}.fallback references "
            f"unknown model_profile {model_profile_id!r}"
        )
    model_profile = model_profiles.get(model_profile_id, {})
    structured_policy = StructuredOutputPolicy.from_model_profile(
        provider=fallback_cfg.get("provider", "mock"),
        model_profile=model_profile,
    )
    return ModelConfig(
        provider=fallback_cfg.get("provider", "mock"),
        model=model_profile.get("model", model_profile_id),
        temperature=model_profile.get("temperature", 0.3),
        max_tokens=model_profile.get("max_tokens"),
        top_p=model_profile.get("top_p", 0.9),
        timeout=model_profile.get("timeout", 10),
        allow_text_tool_fallback=bool(model_profile.get("allow_text_tool_fallback", False)),
        retry_count=int(model_profile.get("retry_count", 1)),
        structured_output_mode=structured_policy.primary_mode.value,
        structured_output_fallback_modes=tuple(
            mode.value for mode in structured_policy.fallback_modes
        ),
        reasoning_level=_reasoning_level(model_profile),
        reasoning_requested=bool(_reasoning_level(model_profile) != "none"),
    )


def _reasoning_level(model_profile: dict[str, Any]) -> str:
    """读取供应商无关的推理意图，不把它误当成已生效能力。"""
    value = model_profile.get("reasoning", "none")
    if isinstance(value, dict):
        value = value.get("level", "none")
    return str(value or "none")


__all__ = ["_resolve_config", "_resolve_fallback_model"]
