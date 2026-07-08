# -*- coding: utf-8 -*-
"""
运行时 agent action pipeline 兼容门面与通用行动适配器。

作者: Mike
创建日期: 2026-07-07
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.runtime.agent_action_pipeline import agent_day_speech
    >>> agent_day_speech(...)
"""

from __future__ import annotations

import logging

from werewolf_agent.evaluation.trace_identity import DecisionIdentity  # noqa: F401
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector  # noqa: F401
from werewolf_agent.runtime.json_extract import (
    extract_first_balanced_json_object as _extract_first_balanced_json_object,  # noqa: F401
)
from werewolf_agent.runtime.reflection_prompt import (
    GOOD_REFLECTION_TEMPLATE,
    WOLF_REFLECTION_TEMPLATE,
    build_reflection_prompt,
)
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS as AGENT_TIMEOUTS  # noqa: F401

# Backward-compatible re-exports from runtime.context (Task 3 extraction).
from werewolf_agent.runtime.context import (
    build_agent_context as build_agent_context,  # noqa: F401
    _SPEECH_STYLE_HINTS as _SPEECH_STYLE_HINTS,  # noqa: F401
    _SHERIFF_SPEECH_STYLE_OVERRIDES as _SHERIFF_SPEECH_STYLE_OVERRIDES,
    _TASK_STYLE_HINTS as _TASK_STYLE_HINTS,
    _get_persona_speech_style as _get_persona_speech_style,  # noqa: F401
    _get_persona_task_style as _get_persona_task_style,
    _action_trace_payload as _action_trace_payload,  # noqa: F401
    _merge_strategy_directive as _merge_strategy_directive,  # noqa: F401
    _inject_skill_output as _inject_skill_output,
)

# Backward-compatible re-exports from runtime.directives package.
from werewolf_agent.runtime.directives import (
    build_hunter_directive as _build_hunter_day_speech_directive,  # noqa: F401
    build_hybrid_directive as _build_hybrid_day_speech_directive,  # noqa: F401
    build_idiot_directive as _build_idiot_day_speech_directive,  # noqa: F401
    build_seer_directive as _build_seer_day_speech_directive,  # noqa: F401
    build_villager_directive as _build_villager_day_speech_directive,  # noqa: F401
    build_witch_directive as _build_witch_day_speech_directive,  # noqa: F401
    build_wolf_day_directive as _build_wolf_day_speech_directive,  # noqa: F401
    build_wolf_night_directive as _build_wolf_night_directive,  # noqa: F401
    build_wolf_vote_directive as _build_wolf_vote_strategy,  # noqa: F401
)
from werewolf_agent.runtime.directives._shared import (
    build_sheriff_silent_directive as _build_sheriff_silent_directive,  # noqa: F401
)
from werewolf_agent.runtime.agent_action_audit import (
    VOTE_BASIS_GUIDANCE,  # noqa: F401
    _audit_context_kwargs as _audit_context_kwargs,  # noqa: F401
    _inject_vote_basis_hint as _inject_vote_basis_hint,  # noqa: F401
    _is_sheriff_silenced as _is_sheriff_silenced,  # noqa: F401
    _seer_credibility_audit_payload as _seer_credibility_audit_payload,  # noqa: F401
)
from werewolf_agent.runtime.agent_reflection_support import (
    _agent_reflection,  # noqa: F401
    _strip_in_game_directives,  # noqa: F401
)
from werewolf_agent.runtime.agent_registry import AgentRegistry, SimpleAgentRegistry  # noqa: F401
from werewolf_agent.runtime.wolf_kill_support import (
    _build_wolf_kill_directive,  # noqa: F401
    _single_wolf_vote,  # noqa: F401
)
from werewolf_agent.runtime.agent_day_actions import (
    agent_day_speech,
    agent_day_vote,
    agent_defense_speech,
    agent_exile_last_words,
    agent_pk_speech,
)
from werewolf_agent.runtime.agent_sheriff_actions import (
    agent_sheriff_endorse,
    agent_sheriff_election_speech,
    agent_sheriff_pick_speech_order,
    agent_sheriff_register,
    agent_sheriff_vote,
    agent_sheriff_withdraw,
)
from werewolf_agent.runtime.agent_special_actions import (
    agent_badge_decision,
    agent_hunter_shot,
    agent_hybrid_choose_master,
    agent_night_seer,
    agent_night_witch,
)
from werewolf_agent.runtime.agent_wolf_actions import (
    agent_wolf_consensus,
    agent_wolf_discussion,
    agent_wolf_team_plan,
)

logger = logging.getLogger(__name__)

# 兼容旧测试和调试入口：反思模板实现已移动到 runtime.reflection_prompt。
_build_reflection_prompt = build_reflection_prompt
_GOOD_REFLECTION_TEMPLATE = GOOD_REFLECTION_TEMPLATE
_WOLF_REFLECTION_TEMPLATE = WOLF_REFLECTION_TEMPLATE


# -- Backward-compatible re-exports from runtime.strategy (Task 2 extraction) --
from werewolf_agent.runtime.strategy import (
    estimate_witch_save_value as _estimate_witch_save_value,  # noqa: F401
    evaluate_hunter_shot_target as _evaluate_hunter_shot_target,  # noqa: F401
    evaluate_hybrid_master_candidates as _evaluate_hybrid_master_candidates,  # noqa: F401
    evaluate_seer_check_value as _evaluate_seer_check_value,  # noqa: F401
    evaluate_wolf_kill_target as _evaluate_wolf_kill_target,  # noqa: F401
    get_wolf_role_assignment as _get_wolf_role_assignment,  # noqa: F401
    has_publicly_claimed_seer as _has_publicly_claimed_seer,  # noqa: F401
)
from werewolf_agent.runtime.strategy.seer import (
    public_seer_claimants as _public_seer_claimants,  # noqa: F401
)


__all__ = [
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
