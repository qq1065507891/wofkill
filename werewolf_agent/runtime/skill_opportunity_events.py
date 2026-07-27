# -*- coding: utf-8 -*-
"""
构造技能机会链的权威审计事件与行动者私有投影。

作者: Project contributors
创建日期: 2026-07-18
修改日期: 2026-07-27

使用示例:
    >>> build_private_skill_event("seer_check_opportunity", actor_id="p01")
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.exposure_audit import (
    is_safe_public_skill_resolution_payload,
)


_PUBLIC_SKILL_RESOLUTION_EVENT_TYPES = frozenset({
    "self_destruct_resolved",
    "hunter_shot_resolved",
})


def is_live_werewolf(game_state: GameState, actor_id: str) -> bool:
    """判断行动者是否仍是可执行自爆的存活狼人。"""
    player = game_state.players.get(actor_id)
    return bool(player and player.alive and player.role == "werewolf")


def _has_authoritative_self_destruct_event(
    game_state: GameState,
    event_type: str,
    *,
    actor_id: str,
    day_number: int,
    opportunity_phase: str,
) -> bool:
    return any(
        event.type == event_type
        and event.visibility is EventVisibility.MODERATOR_ONLY
        and event.payload.get("actor_id") == actor_id
        and event.payload.get("day_number") == day_number
        and event.payload.get("opportunity_phase") == opportunity_phase
        for event in game_state.events
    )


def append_self_destruct_opportunity(
    game_state: GameState,
    *,
    actor_id: str,
    day_number: int,
    opportunity_phase: str,
) -> tuple[GameState, bool]:
    """为存活狼人写入一次可审计的自爆机会，并抵御同节点重入。"""
    if not is_live_werewolf(game_state, actor_id):
        return game_state, False
    if _has_authoritative_self_destruct_event(
        game_state,
        "self_destruct_opportunity",
        actor_id=actor_id,
        day_number=day_number,
        opportunity_phase=opportunity_phase,
    ):
        return game_state, True
    return _append_private_skill_event_unchecked(
        game_state,
        "self_destruct_opportunity",
        actor_id=actor_id,
        day_number=day_number,
        opportunity_phase=opportunity_phase,
    ), True


def append_self_destruct_selected(
    game_state: GameState,
    *,
    actor_id: str,
    day_number: int,
    opportunity_phase: str,
) -> tuple[GameState, bool]:
    """只允许已获机会的存活狼人选择自爆。"""
    if not is_live_werewolf(game_state, actor_id):
        return game_state, False
    if not _has_authoritative_self_destruct_event(
        game_state,
        "self_destruct_opportunity",
        actor_id=actor_id,
        day_number=day_number,
        opportunity_phase=opportunity_phase,
    ):
        return game_state, False
    if _has_authoritative_self_destruct_event(
        game_state,
        "self_destruct_selected",
        actor_id=actor_id,
        day_number=day_number,
        opportunity_phase=opportunity_phase,
    ):
        return game_state, True
    return _append_private_skill_event_unchecked(
        game_state,
        "self_destruct_selected",
        actor_id=actor_id,
        day_number=day_number,
        opportunity_phase=opportunity_phase,
    ), True


def can_select_self_destruct(
    game_state: GameState,
    *,
    actor_id: str,
    day_number: int,
    opportunity_phase: str,
) -> bool:
    """验证当前节点的行动输出确实对应存活狼人的已记录机会。"""
    return is_live_werewolf(game_state, actor_id) and _has_authoritative_self_destruct_event(
        game_state,
        "self_destruct_opportunity",
        actor_id=actor_id,
        day_number=day_number,
        opportunity_phase=opportunity_phase,
    )


def append_self_destruct_declined(
    game_state: GameState,
    *,
    actor_id: str,
    day_number: int,
    opportunity_phase: str,
    reason_code: str,
) -> GameState:
    """为已获得机会但未选择自爆的存活狼人写入一次拒绝。"""
    if (
        not is_live_werewolf(game_state, actor_id)
        or not _has_authoritative_self_destruct_event(
            game_state,
            "self_destruct_opportunity",
            actor_id=actor_id,
            day_number=day_number,
            opportunity_phase=opportunity_phase,
        )
        or _has_authoritative_self_destruct_event(
            game_state,
            "self_destruct_selected",
            actor_id=actor_id,
            day_number=day_number,
            opportunity_phase=opportunity_phase,
        )
        or _has_authoritative_self_destruct_event(
            game_state,
            "self_destruct_declined",
            actor_id=actor_id,
            day_number=day_number,
            opportunity_phase=opportunity_phase,
        )
    ):
        return game_state
    return _append_private_skill_event_unchecked(
        game_state,
        "self_destruct_declined",
        actor_id=actor_id,
        day_number=day_number,
        opportunity_phase=opportunity_phase,
        reason_code=reason_code,
    )


def has_authorized_self_destruct_selection(
    game_state: GameState,
    *,
    actor_id: str,
    day_number: int,
) -> bool:
    """验证当前日存在与机会同阶段匹配的 moderator-only 自爆选择。"""
    for event in game_state.events:
        if (
            event.type != "self_destruct_selected"
            or event.visibility is not EventVisibility.MODERATOR_ONLY
            or event.payload.get("actor_id") != actor_id
            or event.payload.get("day_number") != day_number
        ):
            continue
        phase = event.payload.get("opportunity_phase")
        if isinstance(phase, str) and _has_authoritative_self_destruct_event(
            game_state,
            "self_destruct_opportunity",
            actor_id=actor_id,
            day_number=day_number,
            opportunity_phase=phase,
        ):
            return True
    return False


def has_canonical_self_destruct_resolution(
    game_state: GameState,
    *,
    actor_id: str,
) -> bool:
    """判断行动者是否已经拥有公开的权威自爆结算。"""
    return any(
        event.type == "self_destruct_resolved"
        and event.visibility is EventVisibility.PUBLIC
        and event.payload.get("actor_id") == actor_id
        for event in game_state.events
    )


def build_private_skill_event(
    event_type: str,
    *,
    actor_id: str,
    day_number: int = 0,
    night_number: int = 0,
    **details: Any,
) -> tuple[GameEvent, GameEvent]:
    """返回 moderator-only 权威事件及行动者私有投影。"""
    payload = {
        "actor_id": actor_id,
        "day_number": day_number,
        "night_number": night_number,
        **details,
    }
    authoritative = GameEvent(
        type=event_type,
        payload=payload,
        visibility=EventVisibility.MODERATOR_ONLY,
    )
    actor_view = GameEvent(
        type=f"{event_type}_actor_view",
        payload={**payload, "visibility_actor_id": actor_id},
        visibility=EventVisibility.ACTOR_PRIVATE,
    )
    return authoritative, actor_view


def build_public_skill_resolution(
    event_type: str,
    *,
    actor_id: str,
    target_id: str | None = None,
    public_result: str,
    day_number: int | None = None,
    **_private_details: Any,
) -> GameEvent:
    """构造不携带私有理由或身份真值的公开结算事件。"""
    if event_type not in _PUBLIC_SKILL_RESOLUTION_EVENT_TYPES:
        raise ValueError("public_skill_resolution_event_type")
    payload: dict[str, Any] = {
        "actor_id": actor_id,
        "target_id": target_id,
        "public_result": public_result,
    }
    if day_number is not None:
        payload["day_number"] = day_number
    if not is_safe_public_skill_resolution_payload(payload):
        raise ValueError("unsafe_public_skill_resolution_payload")
    return GameEvent(
        type=event_type,
        payload=payload,
        visibility=EventVisibility.PUBLIC,
    )


def _append_private_skill_event_unchecked(
    game_state: GameState,
    event_type: str,
    *,
    actor_id: str,
    day_number: int = 0,
    night_number: int = 0,
    **details: Any,
) -> GameState:
    """把权威事件和行动者投影作为同一不可分割的审计写入。"""
    return replace(
        game_state,
        events=game_state.events + list(build_private_skill_event(
            event_type,
            actor_id=actor_id,
            day_number=day_number,
            night_number=night_number,
            **details,
        )),
    )


def append_private_skill_event(
    game_state: GameState,
    event_type: str,
    *,
    actor_id: str,
    day_number: int = 0,
    night_number: int = 0,
    **details: Any,
) -> GameState:
    """写入私有技能事件；自爆链统一经幂等机会、选择和拒绝入口。"""
    opportunity_phase = details.get("opportunity_phase")
    if event_type == "self_destruct_opportunity" and isinstance(opportunity_phase, str):
        return append_self_destruct_opportunity(
            game_state,
            actor_id=actor_id,
            day_number=day_number,
            opportunity_phase=opportunity_phase,
        )[0]
    if event_type == "self_destruct_selected" and isinstance(opportunity_phase, str):
        return append_self_destruct_selected(
            game_state,
            actor_id=actor_id,
            day_number=day_number,
            opportunity_phase=opportunity_phase,
        )[0]
    if event_type == "self_destruct_declined" and isinstance(opportunity_phase, str):
        reason_code = details.get("reason_code")
        if isinstance(reason_code, str):
            return append_self_destruct_declined(
                game_state,
                actor_id=actor_id,
                day_number=day_number,
                opportunity_phase=opportunity_phase,
                reason_code=reason_code,
            )
    return _append_private_skill_event_unchecked(
        game_state,
        event_type,
        actor_id=actor_id,
        day_number=day_number,
        night_number=night_number,
        **details,
    )


__all__ = [
    "append_private_skill_event",
    "append_self_destruct_declined",
    "append_self_destruct_opportunity",
    "append_self_destruct_selected",
    "build_private_skill_event",
    "build_public_skill_resolution",
    "can_select_self_destruct",
    "has_authorized_self_destruct_selection",
    "has_canonical_self_destruct_resolution",
    "is_live_werewolf",
]
