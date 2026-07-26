# -*- coding: utf-8 -*-
"""
集中定义玩家 LLM 任务的最低推理级别并校验启动配置。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-26
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.model_gateway.execution_records import ReasoningLevel
from werewolf_agent.model_gateway.router_config import _canonical_provider_name


_MEDIUM_TASKS = {
    "speech", "last_words", "vote", "sheriff_speech", "sheriff_registration",
    "defense_speech", "pk_speech", "discussion_summary",
}
_HIGH_TASKS = {
    "night_action", "deception", "reflection", "wolf_discussion",
    "wolf_team_plan", "hunter_shot", "seer", "witch",
}
_NONE_TASKS = {
    "judge_phase", "judge_death", "judge_vote_calling", "judge_vote_tally",
    "judge_skill_guide", "judge_sheriff", "judge_exile", "rule", "deterministic_judge",
}
_ORDER = {
    ReasoningLevel.NONE: 0,
    ReasoningLevel.LOW: 1,
    ReasoningLevel.MEDIUM: 2,
    ReasoningLevel.HIGH: 3,
}


def minimum_reasoning_level(task_type: str) -> ReasoningLevel:
    """返回任务最低等级；未知玩家任务拒绝隐式无推理执行。"""
    normalized = str(task_type).lower()
    if normalized in _MEDIUM_TASKS:
        return ReasoningLevel.MEDIUM
    if normalized in _HIGH_TASKS:
        return ReasoningLevel.HIGH
    if normalized in _NONE_TASKS:
        return ReasoningLevel.NONE
    raise ValueError(f"unknown player LLM task: {task_type}")


def enforce_minimum_reasoning(task_type: str, configured: str) -> ReasoningLevel:
    """将请求提升到任务最低等级，绝不向下覆盖。"""
    minimum = minimum_reasoning_level(task_type)
    try:
        current = ReasoningLevel(str(configured or "none").lower())
    except ValueError as exc:
        raise ValueError(f"invalid reasoning level {configured!r}") from exc
    return current if _ORDER[current] >= _ORDER[minimum] else minimum


def _profile_level(model_profile: dict[str, Any]) -> ReasoningLevel:
    if _canonical_provider_name(model_profile.get("provider", "")) == "glm":
        return ReasoningLevel.NONE
    raw = model_profile.get("reasoning", "none")
    if isinstance(raw, dict):
        raw = raw.get("level", "none")
    try:
        return ReasoningLevel(str(raw or "none").lower())
    except ValueError as exc:
        raise ValueError(f"invalid reasoning level {raw!r}") from exc


def validate_player_reasoning_profiles(
    *,
    model_profiles: dict[str, dict[str, Any]],
    llm_profiles: dict[str, dict[str, Any]],
    player_assignments: dict[str, str],
) -> None:
    """校验每个玩家 profile 对所有玩家任务均满足最低等级。"""
    for profile_id in sorted(set(player_assignments.values())):
        profile = llm_profiles.get(profile_id, {})
        fallback_raw = profile.get("fallback") or []
        fallbacks = fallback_raw if isinstance(fallback_raw, list) else [fallback_raw]
        for task_type in sorted(_MEDIUM_TASKS | _HIGH_TASKS):
            source = (profile.get("tasks") or {}).get(task_type) or profile.get("default") or {}
            candidates = [source, *fallbacks]
            levels = [
                _profile_level(model_profiles.get(item.get("model_profile", ""), {}))
                for item in candidates
            ]
            minimum = minimum_reasoning_level(task_type)
            if not any(_ORDER[level] >= _ORDER[minimum] for level in levels):
                raise ValueError(
                    f"llm_profile {profile_id!r} task {task_type!r} required "
                    f"{minimum.value!r}, actual {levels[0].value!r}"
                )


def reasoning_capability_satisfies(capability: str, required: str) -> bool:
    """判断候选模型能力是否足以执行请求，禁止通过重标记提升能力。"""
    return _ORDER[ReasoningLevel(capability)] >= _ORDER[ReasoningLevel(required)]


__all__ = [
    "enforce_minimum_reasoning",
    "minimum_reasoning_level",
    "reasoning_capability_satisfies",
    "validate_player_reasoning_profiles",
]
