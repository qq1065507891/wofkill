# -*- coding: utf-8 -*-
"""
集中定义玩家 LLM 任务的最低推理级别并校验启动配置。

作者: Project contributors
创建日期: 2026-07-13
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.model_gateway.execution_records import ReasoningLevel


_MEDIUM_TASKS = {
    "speech", "last_words", "vote", "sheriff_speech", "sheriff_registration",
    "defense_speech", "pk_speech",
}
_HIGH_TASKS = {
    "night_action", "deception", "reflection", "wolf_discussion",
    "wolf_team_plan", "hunter_shot", "seer", "witch", "cognition",
    "cognition_helper", "semantic_repair",
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
        for task_type in sorted(_MEDIUM_TASKS | _HIGH_TASKS):
            source = (profile.get("tasks") or {}).get(task_type) or profile.get("default") or {}
            model_id = source.get("model_profile", "")
            level = _profile_level(model_profiles.get(model_id, {}))
            minimum = ReasoningLevel.MEDIUM
            if _ORDER[level] < _ORDER[minimum]:
                raise ValueError(
                    f"llm_profile {profile_id!r} task {task_type!r} reasoning "
                    f"{level.value!r} is below required {minimum.value!r}"
                )
        fallback = profile.get("fallback") or {}
        if fallback:
            model_id = fallback.get("model_profile", "")
            level = _profile_level(model_profiles.get(model_id, {}))
            if _ORDER[level] < _ORDER[ReasoningLevel.MEDIUM]:
                raise ValueError(
                    f"llm_profile {profile_id!r} fallback reasoning {level.value!r} "
                    "is below required 'medium'"
                )


__all__ = [
    "enforce_minimum_reasoning",
    "minimum_reasoning_level",
    "validate_player_reasoning_profiles",
]
