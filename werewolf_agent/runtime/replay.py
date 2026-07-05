# -*- coding: utf-8 -*-
"""Replay: reconstruct GameState from initial seed + ruleset + event log.
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-05
    使用示例: 内部模块，无对外接口
"""

from __future__ import annotations

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine


def replay_from_events(
    engine: RuleEngine,
    initial_state: GameState,
    events: list[GameEvent],
) -> GameState:
    return engine.reduce_events(initial_state, events)


def extract_event_log(state: GameState) -> list[GameEvent]:
    return list(state.events)
