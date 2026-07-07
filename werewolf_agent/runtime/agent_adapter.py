# -*- coding: utf-8 -*-
"""
运行时 agent adapter 兼容入口。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.runtime.agent_adapter import agent_day_speech
    >>> agent_day_speech(...)
"""

from __future__ import annotations

import sys
from types import ModuleType

from werewolf_agent.runtime import agent_action_pipeline as _action_pipeline
from werewolf_agent.runtime import agent_sheriff_actions as _sheriff_actions
from werewolf_agent.runtime import agent_wolf_actions as _wolf_actions
from werewolf_agent.runtime.agent_action_pipeline import (
    AgentRegistry,
    SimpleAgentRegistry,
    VOTE_BASIS_GUIDANCE,
    _GOOD_REFLECTION_TEMPLATE,
    _WOLF_REFLECTION_TEMPLATE,
    _SPEECH_STYLE_HINTS,
    _SHERIFF_SPEECH_STYLE_OVERRIDES,
    _TASK_STYLE_HINTS,
    _action_trace_payload,
    _agent_reflection,
    _audit_context_kwargs,
    _build_hunter_day_speech_directive,
    _build_hybrid_day_speech_directive,
    _build_idiot_day_speech_directive,
    _build_reflection_prompt,
    _build_sheriff_silent_directive,
    _build_villager_day_speech_directive,
    _build_witch_day_speech_directive,
    _build_wolf_day_speech_directive,
    _build_wolf_kill_directive,
    _build_wolf_night_directive,
    _build_wolf_vote_strategy,
    _estimate_witch_save_value,
    _evaluate_hunter_shot_target,
    _evaluate_hybrid_master_candidates,
    _evaluate_seer_check_value,
    _evaluate_wolf_kill_target,
    _get_persona_speech_style,
    _get_persona_task_style,
    _get_wolf_role_assignment,
    _has_publicly_claimed_seer,
    _inject_skill_output,
    _inject_vote_basis_hint,
    _is_sheriff_silenced,
    _merge_strategy_directive,
    _public_seer_claimants,
    _seer_credibility_audit_payload,
    _single_wolf_vote,
    _strip_in_game_directives,
    build_agent_context,
    logger,
    agent_night_witch,
    agent_night_seer,
    agent_wolf_team_plan,
    agent_wolf_consensus,
    agent_wolf_discussion,
    agent_defense_speech,
    agent_day_speech,
    agent_sheriff_pick_speech_order,
    agent_sheriff_endorse,
    agent_pk_speech,
    agent_day_vote,
    agent_hybrid_choose_master,
    agent_exile_last_words,
    agent_badge_decision,
    agent_hunter_shot,
    agent_sheriff_vote,
    agent_sheriff_register,
    agent_sheriff_withdraw,
    agent_sheriff_election_speech,
)


def __getattr__(name: str):
    """把旧 adapter 路径的兼容符号委托给 action pipeline。"""
    try:
        return getattr(_action_pipeline, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


class _AgentAdapterFacadeModule(ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if hasattr(_action_pipeline, name):
            setattr(_action_pipeline, name, value)
        if hasattr(_sheriff_actions, name):
            setattr(_sheriff_actions, name, value)
        if hasattr(_wolf_actions, name):
            setattr(_wolf_actions, name, value)


sys.modules[__name__].__class__ = _AgentAdapterFacadeModule

__all__ = [
    "AgentRegistry",
    "SimpleAgentRegistry",
    "VOTE_BASIS_GUIDANCE",
    "_GOOD_REFLECTION_TEMPLATE",
    "_WOLF_REFLECTION_TEMPLATE",
    "_SPEECH_STYLE_HINTS",
    "_SHERIFF_SPEECH_STYLE_OVERRIDES",
    "_TASK_STYLE_HINTS",
    "_action_trace_payload",
    "_agent_reflection",
    "_audit_context_kwargs",
    "_build_hunter_day_speech_directive",
    "_build_hybrid_day_speech_directive",
    "_build_idiot_day_speech_directive",
    "_build_reflection_prompt",
    "_build_sheriff_silent_directive",
    "_build_villager_day_speech_directive",
    "_build_witch_day_speech_directive",
    "_build_wolf_day_speech_directive",
    "_build_wolf_kill_directive",
    "_build_wolf_night_directive",
    "_build_wolf_vote_strategy",
    "_estimate_witch_save_value",
    "_evaluate_hunter_shot_target",
    "_evaluate_hybrid_master_candidates",
    "_evaluate_seer_check_value",
    "_evaluate_wolf_kill_target",
    "_get_persona_speech_style",
    "_get_persona_task_style",
    "_get_wolf_role_assignment",
    "_has_publicly_claimed_seer",
    "_inject_skill_output",
    "_inject_vote_basis_hint",
    "_is_sheriff_silenced",
    "_merge_strategy_directive",
    "_public_seer_claimants",
    "_seer_credibility_audit_payload",
    "_single_wolf_vote",
    "_strip_in_game_directives",
    "build_agent_context",
    "logger",
    "agent_night_witch",
    "agent_night_seer",
    "agent_wolf_team_plan",
    "agent_wolf_consensus",
    "agent_wolf_discussion",
    "agent_defense_speech",
    "agent_day_speech",
    "agent_sheriff_pick_speech_order",
    "agent_sheriff_endorse",
    "agent_pk_speech",
    "agent_day_vote",
    "agent_hybrid_choose_master",
    "agent_exile_last_words",
    "agent_badge_decision",
    "agent_hunter_shot",
    "agent_sheriff_vote",
    "agent_sheriff_register",
    "agent_sheriff_withdraw",
    "agent_sheriff_election_speech",
]
