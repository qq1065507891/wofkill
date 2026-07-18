# -*- coding: utf-8 -*-
"""
提供夜晚行动结算与终局提交节点。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-18

使用示例:
    >>> from werewolf_agent.runtime.nodes.night_resolution import resolve_night
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.nodes._shared import (
    RuntimeState,
    logger,
    _judge_broadcast,
    _has_pending_hunter_shot,
    _player_display,
)
from werewolf_agent.runtime.nodes.day_finish import _commit_victory
from werewolf_agent.runtime.skill_opportunity_events import build_private_skill_event

def resolve_night(state: RuntimeState) -> dict[str, Any]:

    engine: RuleEngine = state["engine"]

    gs: GameState = state["game_state"]

    gs, events = engine.resolve_night(

        gs,

        night_number=gs.night_number,

        wolf_kill_target_id=state.get("wolf_kill_target_id"),

        use_antidote=state.get("use_antidote", False),

        poison_target_id=state.get("poison_target_id"),

        seer_target_id=state.get("seer_target_id"),

    )

    seer_trace = state.get("seer_action_trace")

    if seer_trace:

        events = [

            replace(event, payload={**event.payload, "action_trace": seer_trace})

            if event.type == "seer_check" else event

            for event in events

        ]

    if events:

        gs = replace(gs, events=gs.events + events)

    # Log night resolution events

    seer_woke = any(

        event.type == "judge_broadcast" and event.payload.get("phase") == "seer_wake"

        for event in gs.events

    )

    for ev in events:

        if ev.type == "witch_antidote_used":

            target = ev.payload.get("target_id", "?")

            logger.debug(f"  [夜晚结算] {_player_display(state, target)} 被狼人袭击，但被女巫解药救活")

        elif ev.type == "witch_poison_used":

            logger.debug(f"  [夜晚结算] {_player_display(state, ev.payload.get('target_id', '?'))} 被女巫毒杀")

        elif ev.type == "seer_check":

            target = ev.payload.get("target_id", "?")

            alignment = ev.payload.get("alignment", "?")

            gs, _ = _judge_broadcast(

                phase="seer_result",

                message=f"他的身份是{'好人' if alignment == 'good' else '狼人'}",

                gs=gs,

                night_number=gs.night_number,

                visibility="seer_private",

            )

            logger.debug(f"  [夜晚结算] 预言家查验 {_player_display(state, target)}: {'好人' if alignment == 'good' else '狼人'}")

    for seer_event in events:
        if seer_event.type != "seer_check":
            continue
        seer_id = next((
            event.payload.get("actor_id")
            for event in reversed(gs.events)
            if event.type in {"seer_check_selected", "seer_check_repaired"}
            and event.payload.get("night_number") == gs.night_number
        ), None)
        if not isinstance(seer_id, str) or not seer_id:
            seer_id = next((
                player_id for player_id, player in gs.players.items()
                if player.role == "seer"
            ), "")
        if seer_id:
            gs = replace(gs, events=gs.events + list(build_private_skill_event(
                "seer_check_resolved",
                actor_id=seer_id,
                night_number=gs.night_number,
                target_id=seer_event.payload.get("target_id"),
                alignment=seer_event.payload.get("alignment"),
                resolution="checked",
            )))

    if seer_woke:

        gs, _ = _judge_broadcast(

            phase="seer_sleep",

            message="预言家请闭眼",

            gs=gs,

            night_number=gs.night_number,

            visibility="moderator_only",

        )

    # 所有强制死亡反应完成后，在任何白天 Agent 调用前提交终局。
    if not _has_pending_hunter_shot(gs):
        gs = _commit_victory({**state, "game_state": gs})["game_state"]
    return {"game_state": gs}
