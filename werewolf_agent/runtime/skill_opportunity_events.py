# -*- coding: utf-8 -*-
"""
构造技能机会链的权威审计事件与行动者私有投影。

作者: Project contributors
创建日期: 2026-07-18

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
    **_private_details: Any,
) -> GameEvent:
    """构造不携带私有理由或身份真值的公开结算事件。"""
    payload: dict[str, Any] = {
        "actor_id": actor_id,
        "target_id": target_id,
        "public_result": public_result,
    }
    if not is_safe_public_skill_resolution_payload(payload):
        raise ValueError("unsafe_public_skill_resolution_payload")
    return GameEvent(
        type=event_type,
        payload=payload,
        visibility=EventVisibility.PUBLIC,
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


__all__ = [
    "append_private_skill_event",
    "build_private_skill_event",
    "build_public_skill_resolution",
]
