# -*- coding: utf-8 -*-
"""
RuleEngine 的死亡记录、死亡事件和存活目标校验 helper。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-15

使用示例:
    >>> from werewolf_agent.engine.rule_death import apply_death
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from werewolf_agent.core.models import Death, GameEvent, GameState
from werewolf_agent.core.resolution_batches import (
    parse_resolution_batch,
    serialize_resolution_batch,
)


def apply_death(
    state: GameState,
    death: Death,
    *,
    can_leave_last_words_fn: Callable[..., bool],
    can_hunter_shoot_fn: Callable[..., bool],
) -> GameState:
    target = state.players[death.player_id]
    if not target.alive:
        return state
    parsed_batch = parse_resolution_batch(death.resolution_batch)
    can_leave_last_words = death.can_leave_last_words
    if can_leave_last_words is None:
        night_number = state.night_number
        if (
            night_number == 0
            and parsed_batch.batch is not None
            and parsed_batch.batch.phase == "night"
        ):
            night_number = parsed_batch.batch.number
        can_leave_last_words = can_leave_last_words_fn(
            death_reason=death.reason,
            timing=death.timing,
            night_number=night_number,
        )
    triggered_skills = list(death.triggered_skills)
    if target.role == "hunter" and can_hunter_shoot_fn(
        state,
        hunter_id=death.player_id,
        death_reason=death.reason,
    ):
        triggered_skills.append("hunter_shot")
    recorded_death = replace(
        death,
        can_leave_last_words=can_leave_last_words,
        triggered_skills=triggered_skills,
        resolution_batch_parse_failed=(
            death.resolution_batch_parse_failed
            or parsed_batch.batch_parse_failed
        ),
    )
    updated = replace(target, alive=False)
    new_players = {**state.players, death.player_id: updated}
    new_deaths = state.deaths + [recorded_death]
    serialized_batch, serialization_failed = serialize_resolution_batch(
        recorded_death.resolution_batch
    )
    new_events = state.events + [GameEvent(
        type="player_died",
        payload={
            "player_id": recorded_death.player_id,
            "reason": recorded_death.reason,
            "timing": recorded_death.timing,
            "resolution_batch": serialized_batch,
            "resolution_batch_parse_failed": (
                recorded_death.resolution_batch_parse_failed
                or serialization_failed
            ),
            "source_player_id": recorded_death.source_player_id,
            "can_leave_last_words": recorded_death.can_leave_last_words,
            "triggered_skills": recorded_death.triggered_skills,
        },
    )]
    return replace(state, players=new_players, deaths=new_deaths, events=new_events)


def validate_alive_target(state: GameState, target_id: str, label: str) -> None:
    target = state.players.get(target_id)
    if target is None:
        raise ValueError(f"{label}_not_found: {target_id}")
    if not target.alive:
        raise ValueError(f"{label}_not_alive: {target_id}")


__all__ = ["apply_death", "validate_alive_target"]
