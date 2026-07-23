# -*- coding: utf-8 -*-
"""
夜晚阶段节点的兼容 facade。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.nodes.night import enter_night
"""

from __future__ import annotations

from werewolf_agent.runtime.agent_adapter import (
    agent_hybrid_choose_master,
    agent_night_seer,
    agent_night_witch,
    agent_wolf_consensus,
    agent_wolf_discussion,
)
from werewolf_agent.runtime.nodes._shared import (
    AGENT_TIMEOUTS,
    RuntimeState,
    logger,
    _action_audit_events,
    _alive_wolves,
    _allocate_decision_identity,
    _build_wolf_team_plan,
    _dispatch_agent,
    _ensure_runtime_audit_state,
    _find_role,
    _force_wolf_kill,
    _hitl_checkpoint,
    _jb,
    _judge_broadcast,
    _planned_wolf_kill,
    _player_display,
    _player_ids,
    _stable_seed,
)
from werewolf_agent.runtime.nodes.night_entry import (
    enter_night,
    night_hunter_idiot_status,
)
from werewolf_agent.runtime.nodes.night_resolution import resolve_night
from werewolf_agent.runtime.nodes.night_specialists import (
    first_night_hybrid_master,
    night_seer,
    night_witch,
)
from werewolf_agent.runtime.nodes.wolf_night_nodes import (
    _build_fallback_wolf_team_plan,
    _legacy_wolf_consensus,
    wolf_consensus,
    wolf_discussion,
    wolf_team_plan_node,
)

__all__ = [
    "AGENT_TIMEOUTS",
    "RuntimeState",
    "agent_hybrid_choose_master",
    "agent_night_seer",
    "agent_night_witch",
    "agent_wolf_consensus",
    "agent_wolf_discussion",
    "enter_night",
    "first_night_hybrid_master",
    "logger",
    "night_hunter_idiot_status",
    "night_seer",
    "night_witch",
    "resolve_night",
    "wolf_consensus",
    "wolf_discussion",
    "wolf_team_plan_node",
    "_action_audit_events",
    "_alive_wolves",
    "_allocate_decision_identity",
    "_build_fallback_wolf_team_plan",
    "_build_wolf_team_plan",
    "_dispatch_agent",
    "_ensure_runtime_audit_state",
    "_find_role",
    "_force_wolf_kill",
    "_hitl_checkpoint",
    "_jb",
    "_judge_broadcast",
    "_legacy_wolf_consensus",
    "_planned_wolf_kill",
    "_player_display",
    "_player_ids",
    "_stable_seed",
]
