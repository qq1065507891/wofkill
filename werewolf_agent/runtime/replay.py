# -*- coding: utf-8 -*-
"""按 V2 序号或 V1 原始顺序重放事件并重建 GameState。
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-15
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
    if events and all(event.sequence_number is not None for event in events):
        indexed = list(enumerate(events))
        ordered_events = [
            event
            for _, event in sorted(
                indexed,
                key=lambda item: (
                    item[1].sequence_number is None,
                    item[1].sequence_number if item[1].sequence_number is not None else item[0],
                ),
            )
        ]
    else:
        ordered_events = list(events)
    return engine.reduce_events(initial_state, ordered_events)


def extract_event_log(state: GameState) -> list[GameEvent]:
    return list(state.events)
