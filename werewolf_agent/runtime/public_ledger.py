# -*- coding: utf-8 -*-
"""Public information ledger derived from visible game events.
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-14
    使用示例: 内部模块，无对外接口
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.cognition.world_state import _infer_claims_from_text

PUBLIC_EVENT_TYPES = {
    "speech",
    "sheriff_speech",
    "tie_pk_speech",
    "exile_last_words",
    "vote_resolved",
    "badge_transferred",
    "badge_torn",
}

PRIVATE_VISIBILITIES = {
    "werewolf_team_only",
    "moderator_only",
    "seer_private",
    "witch_private",
}


def build_public_ledger(game_state: GameState) -> dict[str, list[dict[str, Any]]]:
    """Build a public ledger from public event payloads only."""
    ledger: dict[str, list[dict[str, Any]]] = {
        "role_claims": [],
        "seer_check_claims": [],
        "badge_flow_claims": [],
        "vote_records": [],
        "last_words": [],
        "badge_events": [],
    }
    for event in game_state.events:
        if not _is_public_event(event):
            continue
        if event.type in {"speech", "sheriff_speech", "tie_pk_speech", "exile_last_words"}:
            _add_speech_items(ledger, event)
        elif event.type == "vote_resolved":
            _add_vote_items(ledger, event)
        elif event.type in {"badge_transferred", "badge_torn"}:
            _add_badge_item(ledger, event)
    return ledger


def build_public_claim_text_ledger(game_state: GameState) -> list[dict[str, Any]]:
    """从完整事件流构建带 speaker 与时间边界的公开 claim 文本账本。"""
    speech_types = {
        "speech", "sheriff_speech", "tie_pk_speech", "sheriff_pk_speech",
        "exile_last_words", "night_death_last_words",
    }
    ledger: list[dict[str, Any]] = []
    for event_index, event in enumerate(game_state.events):
        if event.type not in speech_types or not _is_public_event_compatible(event):
            continue
        payload = event.payload or {}
        speaker = str(
            payload.get("speaker")
            or payload.get("player_id")
            or payload.get("candidate_id")
            or ""
        )
        text = str(payload.get("text") or payload.get("speech") or "")
        if speaker and text:
            ledger.append({
                "event_index": event_index,
                "speaker": speaker,
                "text": text,
            })
    return ledger


def _is_public_event_compatible(event: GameEvent) -> bool:
    return event.payload.get("visibility") not in PRIVATE_VISIBILITIES


def _is_public_event(event: GameEvent) -> bool:
    if event.type not in PUBLIC_EVENT_TYPES:
        return False
    return event.payload.get("visibility") not in PRIVATE_VISIBILITIES


def _add_speech_items(
    ledger: dict[str, list[dict[str, Any]]],
    event: GameEvent,
) -> None:
    speaker = event.payload.get("speaker")
    if not speaker:
        return
    day = event.payload.get("day_number", 0)
    text = str(event.payload.get("text", ""))
    for fact in _infer_claims_from_text(speaker=speaker, text=text, day=day):
        if fact.fact_type == "claimed_role":
            ledger["role_claims"].append({
                "day": day,
                "speaker": speaker,
                "role": fact.value,
                "source_event": event.type,
            })
        elif fact.fact_type == "seer_check_claim":
            ledger["seer_check_claims"].append({
                "day": day,
                "speaker": speaker,
                "target": fact.target_player,
                "result": fact.value,
                "source_event": event.type,
            })
        elif fact.fact_type == "badge_flow_claim":
            targets = list(fact.metadata.get("badge_flow_order", ()))
            if targets:
                ledger["badge_flow_claims"].append({
                    "day": day,
                    "speaker": speaker,
                    "targets": targets,
                    "source_event": event.type,
                })
    if event.type == "exile_last_words":
        ledger["last_words"].append({
            "day": day,
            "speaker": speaker,
            "text": text[:500],
            "source_event": event.type,
        })


def _add_vote_items(
    ledger: dict[str, list[dict[str, Any]]],
    event: GameEvent,
) -> None:
    day = event.payload.get("day_number", 0)
    for vote in event.payload.get("votes", []) or []:
        voter = vote.get("voter")
        target = vote.get("target")
        if not voter or not target:
            continue
        ledger["vote_records"].append({
            "day": day,
            "voter": voter,
            "target": target,
            "reason": str(vote.get("reason", ""))[:200],
            "source_event": event.type,
        })


def _add_badge_item(
    ledger: dict[str, list[dict[str, Any]]],
    event: GameEvent,
) -> None:
    item = {
        "day": event.payload.get("day_number", 0),
        "event": event.type,
        "source_event": event.type,
    }
    if event.type == "badge_transferred":
        item["new_sheriff_id"] = event.payload.get("new_sheriff_id")
    ledger["badge_events"].append(item)
