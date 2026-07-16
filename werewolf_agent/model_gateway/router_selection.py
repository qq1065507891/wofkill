# -*- coding: utf-8 -*-
"""
模型路由器配置选择与 fallback 模型解析。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-16

使用示例:
    >>> _resolve_config(model_profiles={}, llm_profiles={}, player_assignments={}, agent_id="p01", task_type="speech")[0].provider
    'mock'
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.model_gateway.fallback_policy import (
    FallbackRoutePlan,
    build_fallback_routes,
)
from werewolf_agent.model_gateway.reasoning_policy import minimum_reasoning_level
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

    configured_level = _reasoning_level(model_profile)
    enforced_level = minimum_reasoning_level(task_type)
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
        reasoning_level=enforced_level.value,
        reasoning_requested=enforced_level.value != "none",
        reasoning_capability=configured_level,
    )

    fallback_plan = build_fallback_routes(
        config,
        _fallback_candidates(
            model_profiles=model_profiles,
            llm_profiles=llm_profiles,
            llm_profile_id=llm_profile_id,
            required_reasoning_level=enforced_level.value,
        ),
        enforced_level.value,
    )
    fallback_provider = (
        fallback_plan.routes[0].provider if fallback_plan.routes else None
    )

    return config, fallback_provider


def _resolve_fallback_model(
    *,
    model_profiles: dict[str, dict[str, Any]],
    llm_profiles: dict[str, dict[str, Any]],
    llm_profile_id: str,
    required_reasoning_level: str = "none",
    candidate_index: int = 0,
    primary_config: ModelConfig | None = None,
) -> ModelConfig | None:
    """解析 fallback model_profile，缺失引用时快速报错。"""
    candidates = _fallback_candidates(
        model_profiles=model_profiles,
        llm_profiles=llm_profiles,
        llm_profile_id=llm_profile_id,
        required_reasoning_level=required_reasoning_level,
    )
    if primary_config is not None:
        candidates = tuple(
            build_fallback_routes(
                primary_config,
                candidates,
                required_reasoning_level,
            ).routes
        )
    if candidate_index >= len(candidates):
        return None
    return candidates[candidate_index]


def _resolve_fallback_routes(
    *,
    model_profiles: dict[str, dict[str, Any]],
    llm_profiles: dict[str, dict[str, Any]],
    llm_profile_id: str,
    primary_config: ModelConfig,
    required_reasoning_level: str,
) -> FallbackRoutePlan:
    """解析候选后统一交给 route policy 构建不可变执行计划。"""
    candidates = _fallback_candidates(
        model_profiles=model_profiles,
        llm_profiles=llm_profiles,
        llm_profile_id=llm_profile_id,
        required_reasoning_level=required_reasoning_level,
    )
    return build_fallback_routes(
        primary_config,
        candidates,
        required_reasoning_level,
    )


def _fallback_candidates(
    *,
    model_profiles: dict[str, dict[str, Any]],
    llm_profiles: dict[str, dict[str, Any]],
    llm_profile_id: str,
    required_reasoning_level: str,
) -> tuple[ModelConfig, ...]:
    """按声明顺序解析 reasoning 合格的 fallback 配置。"""
    llm_profile = llm_profiles.get(llm_profile_id, {})
    fallback_raw = llm_profile.get("fallback", {})
    if not fallback_raw:
        return ()
    fallback_items = fallback_raw if isinstance(fallback_raw, list) else [fallback_raw]
    capable_items = [
        item for item in fallback_items
        if _level_satisfies(
            _reasoning_level(model_profiles.get(item.get("model_profile", ""), {})),
            required_reasoning_level,
        )
    ]
    candidates: list[ModelConfig] = []
    for fallback_cfg in capable_items:
        candidates.append(_fallback_config(model_profiles, llm_profile_id, fallback_cfg))
    return tuple(candidates)


def _fallback_config(
    model_profiles: dict[str, dict[str, Any]],
    llm_profile_id: str,
    fallback_cfg: dict[str, Any],
) -> ModelConfig:
    """把单个已筛选声明解析为不可变模型配置。"""
    from werewolf_agent.model_gateway.providers.base import ProviderConfigError

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
        reasoning_capability=_reasoning_level(model_profile),
    )


def _reasoning_level(model_profile: dict[str, Any]) -> str:
    """读取供应商无关的推理意图，不把它误当成已生效能力。"""
    if str(model_profile.get("provider", "")).lower() == "glm":
        return "none"
    value = model_profile.get("reasoning", "none")
    if isinstance(value, dict):
        value = value.get("level", "none")
    return str(value or "none")


def _level_satisfies(capability: str, required: str) -> bool:
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    return order.get(capability, -1) >= order.get(required, 99)


__all__ = ["_resolve_config", "_resolve_fallback_model", "_resolve_fallback_routes"]
