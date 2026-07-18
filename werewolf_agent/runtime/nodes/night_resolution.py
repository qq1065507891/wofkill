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

from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameEvent, GameState
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


def _has_valid_v2_or_legacy_identity(event: GameEvent, game_id: str) -> bool:
    """接受完整 V2 或完整旧格式事件，拒绝可伪造的半截身份元数据。"""
    metadata = (
        event.event_id,
        event.sequence_number,
        event.occurred_at,
        event.game_id,
        event.schema_version,
    )
    if all(value is None for value in metadata):
        return True
    return (
        event.schema_version == "2"
        and event.game_id == game_id
        and isinstance(event.sequence_number, int)
        and event.sequence_number >= 0
        and event.event_id == f"{game_id}:e{event.sequence_number:06d}"
        and event.occurred_at is not None
    )


def _canonical_seer_choice_for_result(
    game_state: GameState,
    *,
    target_id: str,
) -> tuple[str, int] | None:
    """从当前夜的权威机会链中找到与引擎查验目标完全一致的真实预言家选择。"""
    for index in range(len(game_state.events) - 1, -1, -1):
        choice = game_state.events[index]
        if (
            choice.type not in {"seer_check_selected", "seer_check_repaired"}
            or choice.visibility is not EventVisibility.MODERATOR_ONLY
            or choice.payload.get("night_number") != game_state.night_number
            or choice.payload.get("target_id") != target_id
            or not _has_valid_v2_or_legacy_identity(choice, game_state.game_id)
        ):
            continue
        actor_id = choice.payload.get("actor_id")
        player = game_state.players.get(actor_id) if isinstance(actor_id, str) else None
        if not player or not player.alive or player.role != "seer":
            continue
        has_linked_opportunity = any(
            event.type == "seer_check_opportunity"
            and event.visibility is EventVisibility.MODERATOR_ONLY
            and event.payload.get("actor_id") == actor_id
            and event.payload.get("night_number") == game_state.night_number
            and isinstance(event.payload.get("legal_targets"), list)
            and target_id in event.payload["legal_targets"]
            and _has_valid_v2_or_legacy_identity(event, game_state.game_id)
            for event in game_state.events[:index]
        )
        if has_linked_opportunity:
            return actor_id, index
    return None


def _has_canonical_seer_resolution(
    game_state: GameState,
    *,
    actor_id: str,
    target_id: str,
) -> bool:
    return any(
        event.type == "seer_check_resolved"
        and event.visibility is EventVisibility.MODERATOR_ONLY
        and event.payload.get("actor_id") == actor_id
        and event.payload.get("night_number") == game_state.night_number
        and event.payload.get("target_id") == target_id
        for event in game_state.events
    )

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
        target_id = seer_event.payload.get("target_id")
        if not isinstance(target_id, str) or not target_id:
            continue
        canonical_choice = _canonical_seer_choice_for_result(
            gs,
            target_id=target_id,
        )
        if canonical_choice is None:
            continue
        seer_id, _choice_index = canonical_choice
        if _has_canonical_seer_resolution(
            gs,
            actor_id=seer_id,
            target_id=target_id,
        ):
            continue
        gs = replace(gs, events=gs.events + list(build_private_skill_event(
            "seer_check_resolved",
            actor_id=seer_id,
            night_number=gs.night_number,
            target_id=target_id,
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
