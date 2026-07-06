# -*- coding: utf-8 -*-
"""
RuleEngine 的预言家、混血儿、女巫、猎人和可见性 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.engine.rule_special_roles import check_alignment
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import (
    Action,
    AlignmentResult,
    GameEvent,
    GameState,
    RuleResult,
    VisibleContext,
)


def check_alignment(
    raw: dict[str, Any],
    state: GameState,
    *,
    target_id: str,
) -> AlignmentResult:
    target = state.players[target_id]
    seer_result = raw["roles"][target.role].get("seer_result", "good")
    return AlignmentResult(alignment=seer_result, role=None)


def choose_master(
    raw: dict[str, Any],
    state: GameState,
    *,
    hybrid_id: str,
    master_id: str,
) -> tuple[GameState, GameEvent]:
    if state.hybrid_master_id is not None:
        raise ValueError("Hybrid has already chosen a master")
    if not state.players[master_id].alive:
        raise ValueError("Hybrid cannot choose a dead player as master")
    master = state.players[master_id]
    master_faction = raw["roles"][master.role]["faction"]
    if master_faction == "special_bound_to_master":
        raise ValueError("Hybrid cannot choose another hybrid as master")
    new_state = replace(
        state,
        hybrid_master_id=master_id,
        hybrid_master_faction=master_faction,
    )
    event = GameEvent(
        type="hybrid_master_chosen",
        payload={"hybrid_id": hybrid_id, "master_id": master_id},
    )
    return new_state, event


def legal_witch_actions(
    raw: dict[str, Any],
    state: GameState,
    *,
    witch_id: str,
    night_number: int,
    wolf_kill_target_id: str | None,
) -> list[Action]:
    actions: list[Action] = [Action(type="no_action")]
    witch_cfg = raw["roles"]["witch"]["abilities"]
    if (
        wolf_kill_target_id is not None
        and not state.antidote_used
        and witch_cfg["antidote"]["can_save_wolf_kill_target"]
    ):
        can_self_save = witch_cfg["antidote"].get("can_self_save", False)
        can_save_first_night = witch_cfg["antidote"].get("can_self_save_first_night", False)
        can_save_self = can_self_save or (can_save_first_night and night_number == 1)
        if wolf_kill_target_id != witch_id or can_save_self:
            actions.append(Action(type="use_antidote", target_id=wolf_kill_target_id))
    if not state.poison_used:
        actions.append(Action(type="use_poison"))
    return actions


def resolve_witch_action(
    raw: dict[str, Any],
    state: GameState,
    *,
    witch_id: str,
    night_number: int,
    wolf_kill_target_id: str | None,
    use_antidote: bool,
    poison_target_id: str | None,
    antidote_target_id: str | None = None,
) -> RuleResult:
    witch_cfg = raw["roles"]["witch"]["abilities"]
    if use_antidote and wolf_kill_target_id is None:
        return RuleResult(accepted=False, error_code="witch_no_wolf_kill_target")
    if use_antidote and wolf_kill_target_id not in state.players:
        return RuleResult(accepted=False, error_code="witch_wolf_kill_target_not_found")
    if use_antidote and not state.players[wolf_kill_target_id].alive:
        return RuleResult(accepted=False, error_code="witch_wolf_kill_target_not_alive")
    if use_antidote and antidote_target_id is not None and antidote_target_id != wolf_kill_target_id:
        return RuleResult(accepted=False, error_code="witch_antidote_target_mismatch")
    if use_antidote and state.antidote_used:
        return RuleResult(accepted=False, error_code="witch_antidote_already_used")
    if poison_target_id is not None and poison_target_id not in state.players:
        return RuleResult(accepted=False, error_code="witch_poison_target_not_found")
    if poison_target_id is not None and not state.players[poison_target_id].alive:
        return RuleResult(accepted=False, error_code="witch_poison_target_not_alive")
    if poison_target_id is not None and state.poison_used:
        return RuleResult(accepted=False, error_code="witch_poison_already_used")
    if not witch_cfg["use_both_potions_same_night"] and use_antidote and poison_target_id is not None:
        return RuleResult(accepted=False, error_code="witch_cannot_use_both_potions_same_night")
    can_self_save = witch_cfg["antidote"].get("can_self_save", False)
    can_save_first_night = witch_cfg["antidote"].get("can_self_save_first_night", False)
    can_save_self = can_self_save or (can_save_first_night and night_number == 1)
    if use_antidote and wolf_kill_target_id == witch_id and not can_save_self:
        return RuleResult(accepted=False, error_code="witch_cannot_self_save")
    return RuleResult(accepted=True)


def can_hunter_shoot(
    raw: dict[str, Any],
    death_reason: str,
) -> bool:
    hunter_cfg = raw["roles"]["hunter"]["abilities"]
    return death_reason in hunter_cfg["can_shoot_on_death_reasons"]


def record_private_intent(
    state: GameState,
    *,
    player_id: str,
    private_intent: dict[str, Any],
) -> GameState:
    new_intents = {**state.private_intents, player_id: private_intent}
    return replace(state, private_intents=new_intents)


def build_visible_context(
    raw: dict[str, Any],
    state: GameState,
    *,
    viewer_id: str,
    view_mode: str,
) -> VisibleContext:
    forbidden = set(raw["information_visibility"]["forbidden_for_player_agents"])
    sections: set[str] = set()
    if view_mode == "player_view":
        sections.add("public_state")
        sections.add("own_private_state")
        if viewer_id in state.private_intents:
            sections.add(f"{viewer_id}.private_intent")
        viewer = state.players.get(viewer_id)
        if viewer:
            role_private = raw["information_visibility"].get("private", {})
            role_sections = role_private.get(viewer.role, [])
            sections.update(role_sections)
        sections -= forbidden
    elif view_mode == "moderator_full":
        sections.add("moderator_full")
        sections.add("all_private_states")
    return VisibleContext(view_mode=view_mode, visible_sections=sections)


__all__ = [
    "build_visible_context",
    "can_hunter_shoot",
    "check_alignment",
    "choose_master",
    "legal_witch_actions",
    "record_private_intent",
    "resolve_witch_action",
]
