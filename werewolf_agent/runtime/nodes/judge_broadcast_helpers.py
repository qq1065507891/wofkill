# -*- coding: utf-8 -*-
"""
运行时节点的裁判广播事件构造与 JudgeAgent 文案调度。

作者: Project contributors
创建日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.runtime.nodes.judge_broadcast_helpers import _judge_broadcast
    >>> _judge_broadcast(phase="day", message="天亮了", gs=game_state)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.nodes.runtime_state import RuntimeState
from werewolf_agent.runtime.timeline import phase_label


def _judge_broadcast(
    *,
    phase: str,
    message: str,
    gs: GameState,
    day_number: int = 0,
    night_number: int = 0,
    extra_payload: dict[str, Any] | None = None,
    visibility: str = "public",
    judge_agent: Any = None,
    judge_llm_enabled: bool = False,
    judge_method: str = "phase",
) -> tuple[GameState, GameEvent]:
    """创建裁判广播事件并追加到 GameState。"""
    final_message = message
    if judge_agent is not None and judge_llm_enabled:
        try:
            llm_msg = _generate_judge_message(
                judge_agent,
                phase=phase,
                fallback=message,
                day_number=day_number,
                night_number=night_number,
                extra_payload=extra_payload,
                judge_method=judge_method,
            )
            if llm_msg:
                final_message = llm_msg
        except Exception:
            pass

    payload = _judge_broadcast_payload(
        phase=phase,
        message=final_message,
        day_number=day_number,
        night_number=night_number,
        visibility=visibility,
        extra_payload=extra_payload,
    )
    event = GameEvent(type="judge_broadcast", payload=payload)
    return replace(gs, events=gs.events + [event]), event


def _generate_judge_message(
    judge_agent: Any,
    *,
    phase: str,
    fallback: str,
    day_number: int = 0,
    night_number: int = 0,
    extra_payload: dict[str, Any] | None = None,
    judge_method: str = "phase",
) -> str:
    """调度 JudgeAgent 对应方法，失败时返回空字符串。"""
    ep = extra_payload or {}

    if judge_method == "vote_calling":
        result = judge_agent.broadcast_vote_calling(
            voter_id=ep.get("voter_id", ""),
            voter_name=ep.get("voter_name", ""),
            candidates=ep.get("candidates", []),
            position=ep.get("position", 1),
            total=ep.get("total", 1),
            day_number=day_number,
            sheriff_weight=ep.get("sheriff_weight", 1.0),
        )
        return _message_or_empty(result)

    if judge_method == "skill_guide":
        result = judge_agent.guide_skill_use(
            role=ep.get("role", ""),
            player_id=ep.get("player_id", ""),
            player_name=ep.get("player_name", ""),
            available_actions=ep.get("available_actions", []),
            context_hints=ep.get("context_hints"),
        )
        return _message_or_empty(result)

    if judge_method == "vote_tally":
        result = judge_agent.announce_vote_tally(
            tally=ep.get("tally", {}),
            player_names=ep.get("player_names", {}),
            sheriff_id=ep.get("sheriff_id"),
            sheriff_weight=ep.get("sheriff_weight", 1.5),
            day_number=day_number,
        )
        return _message_or_empty(result)

    if judge_method == "exile":
        result = judge_agent.announce_exile_result(
            exiled_player_id=ep.get("exiled_player_id"),
            exiled_player_name=ep.get("exiled_player_name", ""),
            reason=ep.get("reason", ""),
            tied_player_ids=ep.get("tied_player_ids"),
            day_number=day_number,
        )
        return _message_or_empty(result)

    if judge_method == "death":
        deaths = ep.get("deaths", [])
        result = judge_agent.broadcast_death_announcement(
            deaths=deaths,
            day_number=day_number,
        )
        return _message_or_empty(result)

    if judge_method == "sheriff":
        result = judge_agent.broadcast_sheriff_result(
            sheriff_id=ep.get("sheriff_id"),
            badge_state=ep.get("badge_state", "none"),
        )
        return _message_or_empty(result)

    public_data = _phase_public_data(ep, day_number=day_number, night_number=night_number)
    result = judge_agent.broadcast_phase(
        phase=phase,
        day_number=day_number,
        night_number=night_number,
        public_data=public_data or None,
    )
    return _message_or_empty(result)


def _jb(
    state: RuntimeState,
    *,
    phase: str,
    message: str,
    gs: GameState | None = None,
    day_number: int = 0,
    night_number: int = 0,
    extra_payload: dict[str, Any] | None = None,
    visibility: str = "public",
    judge_method: str = "phase",
) -> tuple[GameState, GameEvent]:
    """从 RuntimeState 提取 JudgeAgent 后创建裁判广播。"""
    if gs is None:
        gs = state["game_state"]
    gs, event = _judge_broadcast(
        phase=phase,
        message=message,
        gs=gs,
        day_number=day_number,
        night_number=night_number,
        extra_payload=extra_payload,
        visibility=visibility,
        judge_agent=state.get("judge_agent"),
        judge_llm_enabled=state.get("judge_llm_enabled", False),
        judge_method=judge_method,
    )
    state["game_state"] = gs
    return gs, event


def _judge_broadcast_payload(
    *,
    phase: str,
    message: str,
    day_number: int,
    night_number: int,
    visibility: str,
    extra_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phase": phase,
        "message": message,
        "day_number": day_number,
        "night_number": night_number,
        "visibility": visibility,
    }
    if night_number > 0:
        payload["phase_label"] = phase_label("night", night_number)
    elif day_number > 0:
        payload["phase_label"] = phase_label("day", day_number)
    if extra_payload:
        payload.update(extra_payload)
    return payload


def _phase_public_data(
    extra_payload: dict[str, Any],
    *,
    day_number: int,
    night_number: int,
) -> dict[str, Any]:
    public_data = dict(extra_payload)
    if night_number > 0:
        public_data["night_number"] = night_number
    if day_number > 0:
        public_data["day_number"] = day_number
    return public_data


def _message_or_empty(result: Any) -> str:
    return result.message if result and result.message else ""


__all__ = [
    "_generate_judge_message",
    "_jb",
    "_judge_broadcast",
]
