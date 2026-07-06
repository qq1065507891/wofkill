# -*- coding: utf-8 -*-
"""
警长归票节点和归票 adapter。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.nodes.sheriff_endorse import sheriff_endorse
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import agent_sheriff_endorse
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.nodes._shared import (
    RuntimeState,
    logger,
    _action_audit_events,
    _allocate_decision_identity,
    _dispatch_agent,
    _judge_broadcast,
    _player_display,
)


def sheriff_endorse(state: RuntimeState) -> dict[str, Any]:
    """警长放逐投票前私下决定归票目标。"""
    gs: GameState = state["game_state"]
    sheriff_id = gs.sheriff_id
    if not sheriff_id or gs.sheriff_badge_state != "active":
        return {}

    gs, _ = _judge_broadcast(
        phase="vote_start",
        message="讨论结束，现在开始放逐投票。请警长进行归票。",
        gs=gs, day_number=gs.day_number,
        visibility="public",
    )

    decision_identity = _allocate_decision_identity(
        state,
        player_id=sheriff_id,
        phase="sheriff_endorse",
        task_type="vote",
        day_number=gs.day_number,
        night_number=gs.night_number,
    )
    exposure_collector = ModuleExposureAuditCollector()
    result = _dispatch_agent(
        state,
        _sheriff_endorse_adapter,
        sheriff_id,
        timeout_override=120,
        decision_identity=decision_identity,
        exposure_collector=exposure_collector,
    )

    if result:
        endorse_target = result.get("endorse_target", "")
        action_trace = result.get("action_trace")

        if action_trace:
            gs = replace(gs, events=gs.events + _action_audit_events(
                state=state,
                player_id=sheriff_id,
                phase="sheriff_endorse",
                action_trace=action_trace,
                decision_identity=decision_identity,
                exposure_collector=exposure_collector,
                day_number=gs.day_number,
                night_number=gs.night_number,
            ))
        else:
            exposure_collector.flush_events()

        if endorse_target and endorse_target in gs.players:
            gs, _ = _judge_broadcast(
                phase="sheriff_endorse_result",
                message=f"警长{_player_display(state, sheriff_id)}归票{_player_display(state, endorse_target)}",
                gs=gs, day_number=gs.day_number,
                visibility="public",
                extra_payload={"sheriff_id": sheriff_id, "endorse_target": endorse_target},
            )
            logger.debug(
                f"  [警长归票] {_player_display(state, sheriff_id)} → "
                f"{_player_display(state, endorse_target)}"
            )
        else:
            gs, _ = _judge_broadcast(
                phase="sheriff_endorse_result",
                message=f"警长{_player_display(state, sheriff_id)}选择不归票",
                gs=gs, day_number=gs.day_number,
                visibility="public",
            )

        gs = replace(gs, events=gs.events + [GameEvent(
            type="sheriff_endorse",
            payload={
                "speaker": sheriff_id,
                "day_number": gs.day_number,
                "endorse_target": endorse_target,
                "visibility": "public",
            },
        )])
    else:
        gs = replace(gs, events=gs.events + [GameEvent(
            type="sheriff_endorse",
            payload={"speaker": sheriff_id, "day_number": gs.day_number, "text": ""},
        )])

    return {"game_state": gs}


def _sheriff_endorse_adapter(
    state: RuntimeState,
    engine: RuleEngine,
    registry: Any,
    sheriff_id: str,
    *,
    decision_identity: Any = None,
    exposure_collector: Any = None,
    decision_trace_sink: Any = None,
) -> dict[str, Any]:
    """通过 agent_sheriff_endorse 归票，保留现代 build_agent_context adapter 路径。"""
    result = agent_sheriff_endorse(
        state,
        engine,
        registry,
        sheriff_id,
        decision_identity=decision_identity,
        exposure_collector=exposure_collector,
        decision_trace_sink=decision_trace_sink,
    )
    if result is None:
        return {}
    return result


__all__ = ["sheriff_endorse", "_sheriff_endorse_adapter"]
