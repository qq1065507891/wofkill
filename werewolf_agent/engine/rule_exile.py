# -*- coding: utf-8 -*-
"""
RuleEngine 的日间放逐、白痴揭示和狼人自爆 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.engine.rule_exile import resolve_exile
"""

from __future__ import annotations

from collections.abc import Callable

from werewolf_agent.core.models import Death, GameEvent, GameState


def legal_exile_targets(state: GameState) -> list[str]:
    return [
        pid for pid, player in state.players.items()
        if player.alive and not player.exile_immune
    ]


def resolve_exile(
    state: GameState,
    *,
    target_id: str,
    apply_death_fn: Callable[[GameState, Death], GameState],
    apply_idiot_reveal_fn: Callable[[GameState, str], GameState],
) -> tuple[GameState, list[GameEvent]]:
    target = state.players[target_id]
    events: list[GameEvent] = []
    if not target.alive or target.exile_immune:
        return state, events

    death = Death(
        player_id=target_id,
        reason="exile",
        timing="day_vote",
        resolution_batch=f"day_{state.day_number}_vote",
    )
    new_state = apply_death_fn(state, death)
    if target.role == "idiot" and not target.revealed_idiot:
        new_state = apply_idiot_reveal_fn(new_state, target_id)
        events.append(GameEvent(type="idiot_revealed", payload={"player_id": target_id}))
    events.append(GameEvent(
        type="player_exiled",
        payload={"player_id": target_id, "resolution_batch": death.resolution_batch},
    ))
    return new_state, events


def resolve_self_destruct(
    state: GameState,
    *,
    wolf_id: str,
    day_number: int,
    apply_death_fn: Callable[[GameState, Death], GameState],
) -> tuple[GameState, list[GameEvent]]:
    wolf = state.players[wolf_id]
    if not wolf.alive or wolf.role != "werewolf":
        return state, []
    death = Death(
        player_id=wolf_id,
        reason="self_destruct",
        timing="day_discussion",
        resolution_batch=f"day_{day_number}_self_destruct",
    )
    new_state = apply_death_fn(state, death)
    event = GameEvent(
        type="werewolf_self_destructed",
        payload={"player_id": wolf_id, "day_number": day_number},
    )
    return new_state, [event]


__all__ = ["legal_exile_targets", "resolve_exile", "resolve_self_destruct"]
