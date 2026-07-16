# -*- coding: utf-8 -*-
"""
模型路由器配置校验与 provider 名称收集。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-16

使用示例:
    >>> _configured_provider_names({"p": {"provider": "mock"}}, {})
    {'mock'}
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.model_gateway.usage_records import ModelConfig


_KNOWN_PROVIDER_NAMES = {"anthropic", "openai", "glm", "minimax", "mock"}


def _validate_config(
    *,
    model_profiles: dict[str, dict[str, Any]],
    llm_profiles: dict[str, dict[str, Any]],
    player_assignments: dict[str, str],
) -> None:
    """交叉校验玩家、LLM profile、model profile 与 provider 引用。"""
    from werewolf_agent.model_gateway.providers.base import ProviderConfigError
    from werewolf_agent.model_gateway.providers.factory import create_provider_from_env

    for pid, profile_id in player_assignments.items():
        if profile_id not in llm_profiles:
            raise ProviderConfigError(
                f"player {pid!r} references unknown llm_profile {profile_id!r}"
            )

    for profile_id, profile in llm_profiles.items():
        for block_name in ("default", "fallback"):
            raw = profile.get(block_name) or {}
            blocks = raw if isinstance(raw, list) else [raw]
            for block in blocks:
                if not block:
                    continue
                mp_id = block.get("model_profile")
                if mp_id and mp_id not in model_profiles:
                    raise ProviderConfigError(
                        f"llm_profile {profile_id!r}.{block_name} "
                        f"references unknown model_profile {mp_id!r}"
                    )
        for task_type, task_cfg in (profile.get("tasks") or {}).items():
            mp_id = task_cfg.get("model_profile")
            if mp_id and mp_id not in model_profiles:
                raise ProviderConfigError(
                    f"llm_profile {profile_id!r}.tasks.{task_type} "
                    f"references unknown model_profile {mp_id!r}"
                )

    for provider_name in _configured_provider_names(model_profiles, llm_profiles):
        if provider_name.lower() not in _KNOWN_PROVIDER_NAMES:
            raise ProviderConfigError(
                f"provider {provider_name!r} is not registered "
                "(known: anthropic, openai, glm, minimax, mock)"
            )
        # 保留旧实现触碰 factory 的导入副作用。
        _ = create_provider_from_env

    _validate_declared_fallback_routes(model_profiles, llm_profiles)


def _validate_declared_fallback_routes(
    model_profiles: dict[str, dict[str, Any]],
    llm_profiles: dict[str, dict[str, Any]],
) -> None:
    """启动时拒绝声明了 fallback 却不存在合法切换的静态路由图。"""
    from werewolf_agent.model_gateway.fallback_policy import build_fallback_routes
    from werewolf_agent.model_gateway.providers.base import ProviderConfigError
    from werewolf_agent.model_gateway.reasoning_policy import minimum_reasoning_level

    for profile_id, llm_profile in llm_profiles.items():
        fallback_raw = llm_profile.get("fallback")
        if not fallback_raw:
            continue
        fallback_items = (
            fallback_raw if isinstance(fallback_raw, list) else [fallback_raw]
        )
        sources = [("default", llm_profile.get("default") or {})]
        sources.extend((llm_profile.get("tasks") or {}).items())
        for task_type, source in sources:
            if not source:
                continue
            minimum = (
                "none"
                if task_type == "default"
                else minimum_reasoning_level(task_type).value
            )
            primary = _route_config(source, model_profiles)
            candidates = tuple(
                _route_config(item, model_profiles) for item in fallback_items
            )
            plan = build_fallback_routes(primary, candidates, minimum)
            if not plan.routes:
                raise ProviderConfigError(
                    f"fallback_route_unavailable: llm_profile {profile_id!r} "
                    f"task {task_type!r} has no legal fallback route"
                )


def _route_config(
    route: dict[str, Any],
    model_profiles: dict[str, dict[str, Any]],
) -> ModelConfig:
    """构造只供启动门禁使用的最小 ModelConfig。"""
    profile_id = route.get("model_profile", "")
    profile = model_profiles.get(profile_id, {})
    reasoning = profile.get("reasoning", "none")
    if isinstance(reasoning, dict):
        reasoning = reasoning.get("level", "none")
    if str(profile.get("provider", "")).lower() == "glm":
        reasoning = "none"
    return ModelConfig(
        provider=str(route.get("provider", "mock")),
        model=str(profile.get("model", profile_id)),
        reasoning_capability=str(reasoning or "none"),
    )


def _configured_provider_names(
    model_profiles: dict[str, dict[str, Any]],
    llm_profiles: dict[str, dict[str, Any]],
) -> set[str]:
    """收集配置中显式声明的 provider 名称。"""
    providers: set[str] = {
        str(cfg.get("provider", ""))
        for cfg in model_profiles.values()
        if cfg.get("provider")
    }
    for llm_profile in llm_profiles.values():
        default_cfg = llm_profile.get("default", {})
        if default_cfg.get("provider"):
            providers.add(str(default_cfg["provider"]))
        for task_cfg in llm_profile.get("tasks", {}).values():
            if task_cfg.get("provider"):
                providers.add(str(task_cfg["provider"]))
        fallback_raw = llm_profile.get("fallback", {})
        fallback_items = fallback_raw if isinstance(fallback_raw, list) else [fallback_raw]
        for fallback_cfg in fallback_items:
            if fallback_cfg.get("provider"):
                providers.add(str(fallback_cfg["provider"]))
    return providers


__all__ = ["_configured_provider_names", "_validate_config"]
