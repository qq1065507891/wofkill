# -*- coding: utf-8 -*-
"""
日间阶段节点兼容 facade，重新导出拆分后的死亡、讨论、投票和结束节点。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-06
使用示例: 内部模块，无对外接口
"""

from __future__ import annotations

import sys
from types import ModuleType

from werewolf_agent.runtime.agent_adapter import (
    agent_day_speech,
    agent_day_vote,
    agent_exile_last_words,
    agent_sheriff_pick_speech_order,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.nodes import day_deaths as _day_deaths
from werewolf_agent.runtime.nodes import day_discussion as _day_discussion
from werewolf_agent.runtime.nodes import day_finish as _day_finish
from werewolf_agent.runtime.nodes import day_vote as _day_vote
from werewolf_agent.runtime.nodes.day_deaths import (
    _REASON_LABELS,
    _death_reason_label,
    announce_deaths,
    announce_deaths_with_badge_loss,
    exile_last_words,
    night_death_last_words,
)
from werewolf_agent.runtime.nodes.day_discussion import free_discussion
from werewolf_agent.runtime.nodes.day_finish import check_victory, finish_game
from werewolf_agent.runtime.nodes.day_vote import (
    _broadcast_vote_details,
    day_vote,
    resolve_exile,
    resolve_vote,
)
from werewolf_agent.runtime.nodes._shared import _dispatch_agent
from werewolf_agent.runtime.nodes._shared import (
    AGENT_TIMEOUTS,
    RuntimeState,
    logger,
    _action_audit_events,
    _action_trace_event,
    _allocate_decision_identity,
    _ensure_day_incremented,
    _ensure_runtime_audit_state,
    _hitl_checkpoint,
    _jb,
    _judge_broadcast,
    _player_display,
    _public_vote_reason,
    _with_vote_target_in_trace,
)
from werewolf_agent.runtime.sheriff_policy import (
    choose_no_sheriff_speech_order,
    choose_sheriff_led_speech_order,
)
from werewolf_agent.runtime.timeline import phase_label

_PATCHED_DEPENDENCIES = {
    "_dispatch_agent": (_day_deaths, _day_discussion, _day_vote),
    "AGENT_TIMEOUTS": (_day_deaths, _day_discussion, _day_vote),
    "ModuleExposureAuditCollector": (_day_deaths, _day_discussion, _day_vote),
    "agent_day_speech": (_day_discussion,),
    "agent_day_vote": (_day_vote,),
    "agent_exile_last_words": (_day_deaths,),
    "agent_sheriff_pick_speech_order": (_day_discussion,),
    "choose_no_sheriff_speech_order": (_day_discussion,),
    "choose_sheriff_led_speech_order": (_day_discussion,),
    "logger": (_day_deaths, _day_discussion, _day_vote, _day_finish),
    "phase_label": (_day_deaths,),
    "_action_audit_events": (_day_deaths, _day_discussion),
    "_action_trace_event": (_day_vote,),
    "_allocate_decision_identity": (_day_deaths, _day_discussion, _day_vote),
    "_ensure_day_incremented": (_day_deaths,),
    "_ensure_runtime_audit_state": (_day_deaths, _day_vote),
    "_hitl_checkpoint": (_day_deaths,),
    "_jb": (_day_vote,),
    "_judge_broadcast": (_day_deaths, _day_discussion, _day_vote, _day_finish),
    "_player_display": (_day_deaths, _day_discussion, _day_vote),
    "_public_vote_reason": (_day_vote,),
    "_with_vote_target_in_trace": (_day_vote,),
}


class _DayFacadeModule(ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        for module in _PATCHED_DEPENDENCIES.get(name, ()):
            setattr(module, name, value)


sys.modules[__name__].__class__ = _DayFacadeModule

__all__ = [
    "AGENT_TIMEOUTS",
    "ModuleExposureAuditCollector",
    "RuntimeState",
    "_REASON_LABELS",
    "_action_audit_events",
    "_action_trace_event",
    "_allocate_decision_identity",
    "_broadcast_vote_details",
    "_death_reason_label",
    "_dispatch_agent",
    "_ensure_day_incremented",
    "_ensure_runtime_audit_state",
    "_hitl_checkpoint",
    "_jb",
    "_judge_broadcast",
    "_player_display",
    "_public_vote_reason",
    "_with_vote_target_in_trace",
    "agent_day_speech",
    "agent_day_vote",
    "agent_exile_last_words",
    "agent_sheriff_pick_speech_order",
    "announce_deaths",
    "announce_deaths_with_badge_loss",
    "check_victory",
    "choose_no_sheriff_speech_order",
    "choose_sheriff_led_speech_order",
    "day_vote",
    "exile_last_words",
    "finish_game",
    "free_discussion",
    "logger",
    "night_death_last_words",
    "phase_label",
    "resolve_exile",
    "resolve_vote",
]
