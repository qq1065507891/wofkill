# -*- coding: utf-8 -*-
"""
RuleEngine 的夜晚结算 helper。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-15

使用示例:
    >>> from werewolf_agent.engine.rule_night import resolve_night
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import Death, GameEvent, GameState
from werewolf_agent.core.resolution_batches import ResolutionBatchV2


def resolve_night(
    raw: dict[str, Any],
    state: GameState,
    *,
    night_number: int,
    wolf_kill_target_id: str | None,
    resolve_witch_action_fn: Callable[..., Any],
    validate_alive_target_fn: Callable[[GameState, str, str], None],
    check_alignment_fn: Callable[..., Any],
    apply_death_fn: Callable[[GameState, Death], GameState],
    use_antidote: bool = False,
    poison_target_id: str | None = None,
    seer_target_id: str | None = None,
) -> tuple[GameState, list[GameEvent]]:
    events: list[GameEvent] = []
    deaths: list[Death] = []
    antidote_used = state.antidote_used
    poison_used = state.poison_used
    witch_id = next(
        (pid for pid, player in state.players.items() if player.role == "witch" and player.alive),
        None,
    )
    if use_antidote or poison_target_id is not None:
        if witch_id is None:
            raise ValueError("witch_not_available")
        witch_result = resolve_witch_action_fn(
            state,
            witch_id=witch_id,
            night_number=night_number,
            wolf_kill_target_id=wolf_kill_target_id,
            use_antidote=use_antidote,
            poison_target_id=poison_target_id,
        )
        if not witch_result.accepted:
            raise ValueError(witch_result.error_code or "witch_action_invalid")

    saved_by_antidote = False
    if wolf_kill_target_id is not None:
        validate_alive_target_fn(state, wolf_kill_target_id, "wolf_kill_target")
        wolf_death = Death(
            player_id=wolf_kill_target_id,
            reason="wolf_kill",
            timing="night",
            resolution_batch=ResolutionBatchV2("night", night_number, "wolf_kill"),
        )
        if use_antidote and not antidote_used:
            witch_cfg = raw["roles"]["witch"]
            antidote_cfg = witch_cfg["abilities"].get("antidote", {})
            can_self_save = antidote_cfg.get("can_self_save", False)
            can_save_first_night = antidote_cfg.get("can_self_save_first_night", False)
            can_save = (
                wolf_kill_target_id != witch_id
                or can_self_save
                or (can_save_first_night and night_number == 1)
            )
            if witch_id is not None and can_save:
                saved_by_antidote = True
                antidote_used = True
                events.append(GameEvent(
                    type="witch_antidote_used",
                    payload={"target_id": wolf_kill_target_id, "visibility": "witch_private"},
                ))
        if not saved_by_antidote:
            deaths.append(wolf_death)

    witch_cfg = raw["roles"]["witch"]["abilities"]
    use_both = witch_cfg.get("use_both_potions_same_night", False)
    if poison_target_id is not None and not poison_used:
        if not saved_by_antidote or use_both:
            validate_alive_target_fn(state, poison_target_id, "poison_target")
            poison_used = True
            deaths.append(Death(
                player_id=poison_target_id,
                reason="witch_poison",
                timing="night",
                resolution_batch=ResolutionBatchV2("night", night_number, "witch_poison"),
            ))
            events.append(GameEvent(
                type="witch_poison_used",
                payload={"target_id": poison_target_id, "visibility": "witch_private"},
            ))

    if seer_target_id is not None:
        seer_id = next(
            (pid for pid, player in state.players.items() if player.role == "seer" and player.alive),
            None,
        )
        if seer_id is not None:
            alignment_result = check_alignment_fn(state, target_id=seer_target_id)
            events.append(GameEvent(
                type="seer_check",
                payload={
                    "target_id": seer_target_id,
                    "alignment": alignment_result.alignment,
                    "night_number": night_number,
                    "visibility": "seer_only",
                },
            ))

    new_state = state
    for death in deaths:
        new_state = apply_death_fn(new_state, death)

    new_state = replace(new_state, antidote_used=antidote_used, poison_used=poison_used)
    return new_state, events


__all__ = ["resolve_night"]
