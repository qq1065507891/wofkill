# -*- coding: utf-8 -*-
"""
提供夜晚阶段入口和猎人/白痴状态确认节点。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.nodes.night_entry import enter_night
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.nodes._shared import (
    RuntimeState,
    logger,
    _find_role,
    _hitl_checkpoint,
    _judge_broadcast,
    _player_display,
)
from werewolf_agent.runtime.timeline import phase_label

def enter_night(state: RuntimeState) -> dict[str, Any]:

    gs: GameState = state["game_state"]

    n = gs.night_number + 1

    label = phase_label("night", n)

    gs, _ = _judge_broadcast(phase="enter_night", message=f"{label}：天黑请闭眼", gs=gs, night_number=n)

    gs = replace(gs, phase="night", night_number=n,

                 events=gs.events + [GameEvent(type="enter_night", payload={"night": n})])

    alive = [pid for pid, p in gs.players.items() if p.alive]

    logger.debug(f"\n{'='*60}")

    logger.debug(f"  【{label}】天黑请闭眼 (存活: {len(alive)}人)")

    logger.debug(f"{'='*60}")

    _hitl_checkpoint(state, "enter_night", "after")

    return {"game_state": gs}



def night_hunter_idiot_status(state: RuntimeState) -> dict[str, Any]:

    """First night only: confirm hunter and idiot are alive for moderator audit.

    Produces no public output; event is moderator/private visibility only."""

    gs: GameState = state["game_state"]

    hunter_id = _find_role(gs, "hunter")

    idiot_id = _find_role(gs, "idiot")

    if hunter_id:

        gs, _ = _judge_broadcast(

            phase="hunter_status",

            message=f"猎人{_player_display(state, hunter_id)}请确认开枪状态",

            gs=gs, night_number=gs.night_number,

            visibility="moderator_only",

        )

        logger.debug(f"  [法官] 猎人{_player_display(state, hunter_id)}请确认开枪状态")

    if idiot_id and gs.night_number == 1:

        gs, _ = _judge_broadcast(

            phase="idiot_status",

            message=f"白痴{_player_display(state, idiot_id)}请确认身份",

            gs=gs, night_number=gs.night_number,

            visibility="moderator_only",

        )

        logger.debug(f"  [法官] 白痴{_player_display(state, idiot_id)}请确认身份")

    if gs.night_number != 1:

        return {}

    event = GameEvent(

        type="hunter_idiot_status_confirmed",

        payload={

            "night_number": 1,

            "hunter_id": hunter_id,

            "idiot_id": idiot_id,

            "visibility": "moderator_only",

        },

    )

    gs = replace(gs, events=gs.events + [event])

    return {"game_state": gs}
