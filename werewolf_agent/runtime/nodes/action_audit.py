# -*- coding: utf-8 -*-
"""
运行时节点的行动轨迹审计、投票私有审计和决策身份 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.nodes.action_audit import _allocate_decision_identity
    >>> _allocate_decision_identity(state, player_id="p01", phase="day", task_type="vote", day_number=1, night_number=0)
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameEvent
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.nodes.runtime_state import RuntimeState
from werewolf_agent.runtime.timeline import detect_timeline_confusion


def _ensure_runtime_audit_state(state: RuntimeState) -> RuntimeState:
    if not isinstance(state.get("action_index_by_game"), dict):
        state["action_index_by_game"] = {}
    if not isinstance(state.get("pending_exposure_events_by_trace"), dict):
        state["pending_exposure_events_by_trace"] = {}
    return state


def _allocate_decision_identity(
    state: RuntimeState,
    *,
    player_id: str,
    phase: str,
    task_type: str,
    day_number: int,
    night_number: int,
) -> DecisionIdentity:
    gs = state["game_state"]
    next_index_by_game = _ensure_runtime_audit_state(state)["action_index_by_game"]
    action_index = int(next_index_by_game.get(gs.game_id, 0))
    next_index_by_game[gs.game_id] = action_index + 1
    return DecisionIdentity(
        game_id=gs.game_id,
        player_id=player_id,
        phase=phase,
        day_number=day_number,
        night_number=night_number,
        task_type=task_type,
        action_index=action_index,
    )


def _action_trace_event(
    *,
    player_id: str,
    phase: str,
    action_trace: dict[str, Any],
    day_number: int = 0,
    night_number: int = 0,
    decision_identity: DecisionIdentity | None = None,
) -> GameEvent:
    audit_text_parts: list[str] = []
    raw_text = action_trace.get("raw_text")
    if raw_text:
        audit_text_parts.append(str(raw_text))
    parsed_action = action_trace.get("parsed_action") or {}
    if isinstance(parsed_action, dict):
        for key in ("reason", "speech_text", "private_reason"):
            value = parsed_action.get(key)
            if value:
                audit_text_parts.append(str(value))
    timeline_confusion = detect_timeline_confusion("\n".join(audit_text_parts))
    payload = {
        "player_id": decision_identity.player_id if decision_identity else player_id,
        "phase": decision_identity.phase if decision_identity else phase,
        "day_number": decision_identity.day_number if decision_identity else day_number,
        "night_number": decision_identity.night_number if decision_identity else night_number,
        "visibility": "moderator_only",
        "action_trace": action_trace,
        "timeline_confusion": timeline_confusion,
    }
    if decision_identity is not None:
        payload.update({
            "trace_id": decision_identity.trace_id(),
            "game_id": decision_identity.game_id,
            "task_type": decision_identity.task_type,
            "action_index": decision_identity.action_index,
        })
    if phase == "vote":
        payload.update(_private_vote_audit_payload(action_trace))

    return GameEvent(type="action_trace_audit", payload=payload)


def _action_audit_events(
    *,
    state: RuntimeState,
    player_id: str,
    phase: str,
    action_trace: dict[str, Any],
    decision_identity: DecisionIdentity | None,
    exposure_collector: ModuleExposureAuditCollector | None,
    day_number: int = 0,
    night_number: int = 0,
) -> list[GameEvent]:
    event = _action_trace_event(
        player_id=player_id,
        phase=phase,
        action_trace=action_trace,
        day_number=day_number,
        night_number=night_number,
        decision_identity=decision_identity,
    )
    exposure_events = exposure_collector.flush_events() if exposure_collector else []
    return [*exposure_events, event]


def _private_vote_audit_payload(action_trace: dict[str, Any]) -> dict[str, Any]:
    parsed = action_trace.get("parsed_action") or {}
    if not isinstance(parsed, dict):
        parsed = {}
    target = (
        parsed.get("target_id")
        or parsed.get("target")
        or action_trace.get("target_id")
        or action_trace.get("target")
    )
    thought = {
        "target": target,
        "public_reason": str(parsed.get("reason") or action_trace.get("reason") or "")[:300],
        "standing_with_seer": str(parsed.get("standing_with_seer") or "")[:100],
        "suspect_reason": str(parsed.get("suspect_reason") or "")[:300],
        "not_voting_reason": str(parsed.get("not_voting_reason") or "")[:300],
        "private_reason": str(parsed.get("private_reason") or "")[:500],
    }
    return {
        "vote_target": target,
        "private_vote_thought": thought,
    }


def _public_vote_reason(action_trace: dict[str, Any] | None) -> str:
    if not action_trace:
        return ""
    parsed = action_trace.get("parsed_action") or {}
    reason = (
        parsed.get("reason")
        or action_trace.get("reason")
        or action_trace.get("fallback_reason")
        or ""
    )
    return str(reason)[:200]


def _with_vote_target_in_trace(
    action_trace: dict[str, Any],
    target_id: str,
) -> dict[str, Any]:
    parsed = action_trace.get("parsed_action")
    if isinstance(parsed, dict):
        return {
            **action_trace,
            "target_id": target_id,
            "parsed_action": {**parsed, "target_id": parsed.get("target_id") or target_id},
        }
    return {**action_trace, "target_id": target_id}


__all__ = [
    "_action_audit_events",
    "_action_trace_event",
    "_allocate_decision_identity",
    "_ensure_runtime_audit_state",
    "_private_vote_audit_payload",
    "_public_vote_reason",
    "_with_vote_target_in_trace",
]
