# -*- coding: utf-8 -*-
"""
集中生成狼人团队决策事件及其稳定的语义 trace 身份。

作者: Project contributors
创建日期: 2026-07-18

使用示例:
    >>> event = new_wolf_decision_event(
    ...     game_state,
    ...     "wolf_kill_selected",
    ...     {"target_id": "p03"},
    ...     visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
    ...     decision_kind=WOLF_KILL_EXPLICIT_STATE,
    ... )
"""

from __future__ import annotations

from typing import Any, Mapping

from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime.event_metadata import new_game_event


WOLF_KILL_LEGACY_AGENT = "wolf_kill_selected:legacy_agent"
WOLF_KILL_EXPLICIT_STATE = "wolf_kill_selected:explicit_state"
WOLF_KILL_FORCED_FALLBACK = "wolf_kill_selected:forced_fallback"

_STANCE_STATUSES = frozenset({"majority", "single_wolf"})
_STANCE_PLAN_KEYS = {
    "night_kill_primary": "primary",
    "night_kill_backup": "backup",
}


def wolf_stance_kill_decision_kind(
    *,
    consensus_status: str,
    plan_key: str,
) -> str:
    """把结构化立场的授权方式编码进稳定决策身份。"""
    if consensus_status not in _STANCE_STATUSES:
        raise ValueError(
            f"unsupported wolf consensus status: {consensus_status}"
        )
    try:
        priority = _STANCE_PLAN_KEYS[plan_key]
    except KeyError as exc:
        raise ValueError(f"unsupported wolf plan key: {plan_key}") from exc
    return f"wolf_kill_selected:stance:{consensus_status}:{priority}"


def wolf_decision_trace_id(
    game_state: GameState,
    *,
    decision_kind: str,
    action_index: int = 0,
) -> str:
    """按局、夜次和决策语义生成可重复计算的狼人团队 trace。"""
    if not decision_kind:
        raise ValueError("wolf decision kind must not be empty")
    return DecisionIdentity(
        game_id=game_state.game_id,
        player_id="werewolf_team",
        phase="wolf_consensus",
        day_number=game_state.day_number,
        night_number=game_state.night_number,
        task_type=decision_kind,
        action_index=action_index,
    ).trace_id()


def new_wolf_decision_event(
    game_state: GameState,
    event_type: str,
    payload: Mapping[str, Any],
    *,
    visibility: EventVisibility,
    decision_kind: str | None = None,
    action_index: int = 0,
) -> GameEvent:
    """创建带 V2 元数据和稳定狼人决策 trace 的生产事件。"""
    return new_game_event(
        game_state,
        event_type,
        payload,
        visibility=visibility,
        trace_id=wolf_decision_trace_id(
            game_state,
            decision_kind=decision_kind or event_type,
            action_index=action_index,
        ),
    )


__all__ = [
    "WOLF_KILL_EXPLICIT_STATE",
    "WOLF_KILL_FORCED_FALLBACK",
    "WOLF_KILL_LEGACY_AGENT",
    "new_wolf_decision_event",
    "wolf_decision_trace_id",
    "wolf_stance_kill_decision_kind",
]
