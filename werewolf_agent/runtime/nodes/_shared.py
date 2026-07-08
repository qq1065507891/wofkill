# -*- coding: utf-8 -*-
"""
运行时图节点共享 helper 的兼容 facade。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.nodes._shared import RuntimeState
    >>> from werewolf_agent.runtime.nodes._shared import _build_wolf_team_plan
"""

from __future__ import annotations

import sys
from types import ModuleType

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.nodes import action_audit as _action_audit
from werewolf_agent.runtime.nodes import judge_broadcast_helpers as _judge_broadcast_helpers
from werewolf_agent.runtime.nodes import node_helpers as _node_helpers
from werewolf_agent.runtime.nodes import runtime_state as _runtime_state
from werewolf_agent.runtime.nodes.action_audit import (
    _action_audit_events,
    _action_trace_event,
    _allocate_decision_identity,
    _ensure_runtime_audit_state,
    _private_vote_audit_payload,
    _public_vote_reason,
    _with_vote_target_in_trace,
)
from werewolf_agent.runtime.nodes.node_helpers import (
    logger,
    _agent_timeout,
    _alive_non_wolves,
    _alive_wolves,
    _build_wolf_team_plan,
    _call_agent,
    _deaths_already_announced,
    _dispatch_agent,
    _ensure_day_incremented,
    _find_role,
    _first_alive_target,
    _force_wolf_kill,
    _generate_judge_message,
    _hitl_checkpoint,
    _jb,
    _judge_broadcast,
    _needs_sheriff_before_deaths,
    _planned_wolf_kill,
    _player_display,
    _player_ids,
    _sheriff_died_this_batch,
    _timer_expired,
)
from werewolf_agent.runtime.nodes.runtime_state import (
    RULESET_PATH,
    RuntimeState,
    _new_engine,
    _stable_seed,
)
from werewolf_agent.runtime.timers import timed_call
from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS
from werewolf_agent.runtime.timeline import detect_timeline_confusion, phase_label


_PATCHED_DEPENDENCIES = {
    "RULESET_PATH": (_runtime_state,),
    "RuleEngine": (_runtime_state,),
    "_stable_seed": (_runtime_state, _node_helpers),
    "detect_timeline_confusion": (_action_audit,),
    "logger": (_node_helpers,),
    "phase_label": (_node_helpers, _judge_broadcast_helpers),
    "timed_call": (_node_helpers,),
}


class _SharedFacadeModule(ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        for module in _PATCHED_DEPENDENCIES.get(name, ()):
            setattr(module, name, value)


sys.modules[__name__].__class__ = _SharedFacadeModule


__all__ = [
    "AGENT_TIMEOUTS",
    "DecisionIdentity",
    "GameEvent",
    "GameState",
    "ModuleExposureAuditCollector",
    "RULESET_PATH",
    "RuleEngine",
    "RuntimeState",
    "detect_timeline_confusion",
    "logger",
    "phase_label",
    "timed_call",
    "_action_audit_events",
    "_action_trace_event",
    "_agent_timeout",
    "_alive_non_wolves",
    "_alive_wolves",
    "_allocate_decision_identity",
    "_build_wolf_team_plan",
    "_call_agent",
    "_deaths_already_announced",
    "_dispatch_agent",
    "_ensure_day_incremented",
    "_ensure_runtime_audit_state",
    "_find_role",
    "_first_alive_target",
    "_force_wolf_kill",
    "_generate_judge_message",
    "_hitl_checkpoint",
    "_jb",
    "_judge_broadcast",
    "_needs_sheriff_before_deaths",
    "_new_engine",
    "_planned_wolf_kill",
    "_player_display",
    "_player_ids",
    "_private_vote_audit_payload",
    "_public_vote_reason",
    "_sheriff_died_this_batch",
    "_stable_seed",
    "_timer_expired",
    "_with_vote_target_in_trace",
]
