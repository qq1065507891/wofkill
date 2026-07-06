# -*- coding: utf-8 -*-
"""
警长节点兼容 facade，重新导出拆分后的报名、退水、投票、发言和归票节点。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.nodes.sheriff import sheriff_vote
"""

from __future__ import annotations

import sys
from types import ModuleType

from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import (
    agent_sheriff_election_speech,
    agent_sheriff_endorse,
    agent_sheriff_register,
    agent_sheriff_vote,
    agent_sheriff_withdraw,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.nodes import sheriff_endorse as _endorse
from werewolf_agent.runtime.nodes import sheriff_registration as _registration
from werewolf_agent.runtime.nodes import sheriff_speech as _speech
from werewolf_agent.runtime.nodes import sheriff_vote as _vote
from werewolf_agent.runtime.nodes._shared import (
    AGENT_TIMEOUTS,
    RuntimeState,
    logger,
    _action_audit_events,
    _allocate_decision_identity,
    _dispatch_agent,
    _ensure_day_incremented,
    _judge_broadcast,
    _player_display,
    _stable_seed,
)
from werewolf_agent.runtime.nodes.sheriff_endorse import (
    _sheriff_endorse_adapter,
    sheriff_endorse,
)
from werewolf_agent.runtime.nodes.sheriff_registration import (
    sheriff_first_day_entry,
    sheriff_registration,
    sheriff_withdraw,
)
from werewolf_agent.runtime.nodes.sheriff_speech import sheriff_speech
from werewolf_agent.runtime.nodes.sheriff_vote import sheriff_vote
from werewolf_agent.runtime.sheriff_policy import (
    choose_no_sheriff_speech_order,
    choose_sheriff_led_speech_order,
    eligible_sheriff_voters,
    filter_sheriff_votes_to_eligible,
    is_all_players_on_sheriff,
)


_PATCHED_DEPENDENCIES = {
    "AGENT_TIMEOUTS": (_speech,),
    "ModuleExposureAuditCollector": (_registration, _vote, _speech, _endorse),
    "agent_sheriff_election_speech": (_speech,),
    "agent_sheriff_endorse": (_endorse,),
    "agent_sheriff_register": (_registration,),
    "agent_sheriff_vote": (_vote,),
    "agent_sheriff_withdraw": (_registration,),
    "choose_no_sheriff_speech_order": (_vote,),
    "choose_sheriff_led_speech_order": (_vote,),
    "eligible_sheriff_voters": (_vote,),
    "filter_sheriff_votes_to_eligible": (_vote,),
    "is_all_players_on_sheriff": (_vote, _speech),
    "logger": (_registration, _vote, _speech, _endorse),
    "_action_audit_events": (_registration, _vote, _speech, _endorse),
    "_allocate_decision_identity": (_registration, _vote, _speech, _endorse),
    "_dispatch_agent": (_registration, _vote, _speech, _endorse),
    "_ensure_day_incremented": (_registration,),
    "_judge_broadcast": (_registration, _vote, _speech, _endorse),
    "_player_display": (_registration, _vote, _speech, _endorse),
    "_sheriff_endorse_adapter": (_endorse,),
    "_stable_seed": (_speech,),
}


class _SheriffFacadeModule(ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        for module in _PATCHED_DEPENDENCIES.get(name, ()):
            setattr(module, name, value)


sys.modules[__name__].__class__ = _SheriffFacadeModule


__all__ = [
    "AGENT_TIMEOUTS",
    "ModuleExposureAuditCollector",
    "RuleEngine",
    "RuntimeState",
    "_action_audit_events",
    "_allocate_decision_identity",
    "_dispatch_agent",
    "_ensure_day_incremented",
    "_judge_broadcast",
    "_player_display",
    "_sheriff_endorse_adapter",
    "_stable_seed",
    "agent_sheriff_election_speech",
    "agent_sheriff_endorse",
    "agent_sheriff_register",
    "agent_sheriff_vote",
    "agent_sheriff_withdraw",
    "choose_no_sheriff_speech_order",
    "choose_sheriff_led_speech_order",
    "eligible_sheriff_voters",
    "filter_sheriff_votes_to_eligible",
    "is_all_players_on_sheriff",
    "logger",
    "sheriff_endorse",
    "sheriff_first_day_entry",
    "sheriff_registration",
    "sheriff_speech",
    "sheriff_vote",
    "sheriff_withdraw",
]
