# -*- coding: utf-8 -*-
"""从 V1/V2 可见事件构建公开事实、玩家声明与冲突账本。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-27
使用示例: 内部模块，无对外接口
"""

from __future__ import annotations

import re
from typing import Any

from werewolf_agent.cognition.world_state import _infer_claims_from_text
from werewolf_agent.core.event_visibility import EventVisibility, event_visibility
from werewolf_agent.core.models import GameEvent, GameState

PUBLIC_EVENT_TYPES = {
    "speech",
    "sheriff_speech",
    "tie_pk_speech",
    "exile_last_words",
    "vote_resolved",
    "badge_transferred",
    "badge_torn",
    "hunter_shot_resolved",
}

PRIVATE_VISIBILITIES = {
    "werewolf_team_only",
    "moderator_only",
    "seer_private",
    "witch_private",
}

_ACTION_CLAIM_EVENT_TYPES = {
    "speech",
    "sheriff_speech",
    "tie_pk_speech",
    "sheriff_pk_speech",
    "exile_last_words",
    "night_death_last_words",
}

_LAST_WORD_EVENT_TYPES = {"exile_last_words", "night_death_last_words"}

_ACTION_CLAIM_PATTERNS = (
    ("hunter_shot", re.compile(r"(?:开枪|带走)\s*(p\d{2})(?![A-Za-z0-9_])")),
    ("witch_antidote", re.compile(r"(?:用解药救|解药救)了?\s*(p\d{2})(?![A-Za-z0-9_])")),
)


def build_public_ledger(game_state: GameState) -> dict[str, list[dict[str, Any]]]:
    """仅从公开事件构建稳定结构的公开账本。"""
    ledger: dict[str, list[dict[str, Any]]] = {
        "role_claims": [],
        "seer_check_claims": [],
        "badge_flow_claims": [],
        "vote_records": [],
        "last_words": [],
        "badge_events": [],
        "action_claims": [],
        "confirmed_actions": [],
        "claim_conflicts": [],
    }
    for event in game_state.events:
        if (
            event.type in _ACTION_CLAIM_EVENT_TYPES
            and _is_public_event_compatible(event)
        ):
            _add_action_claims(ledger, event)
        if not _is_public_event(event):
            continue
        if event.type in {"speech", "sheriff_speech", "tie_pk_speech", "exile_last_words"}:
            _add_speech_items(ledger, event)
        elif event.type == "vote_resolved":
            _add_vote_items(ledger, event)
        elif event.type in {"badge_transferred", "badge_torn"}:
            _add_badge_item(ledger, event)
        elif event.type == "hunter_shot_resolved":
            _add_confirmed_action(ledger, event)
    _add_claim_conflicts(ledger)
    return ledger


def build_claim_action_audit(game_state: GameState) -> list[dict[str, Any]]:
    """以 moderator-only 视角核验公开行动声明。"""
    public_claims = build_public_ledger(game_state)["action_claims"]
    engine_actions = _all_engine_action_evidence(game_state)
    return [
        {
            **claim,
            "status": _claim_action_status(claim, engine_actions),
            "visibility": "moderator_only",
        }
        for claim in public_claims
    ]


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
    return event_visibility(event) is EventVisibility.PUBLIC


def _is_public_event(event: GameEvent) -> bool:
    if event.type not in PUBLIC_EVENT_TYPES:
        return False
    return event_visibility(event) is EventVisibility.PUBLIC


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


def _add_action_claims(
    ledger: dict[str, list[dict[str, Any]]],
    event: GameEvent,
) -> None:
    speaker = event.payload.get("speaker")
    if not speaker:
        return
    day = event.payload.get("day_number", 0)
    text = str(event.payload.get("text", ""))
    support_kind = (
        "last_words" if event.type in _LAST_WORD_EVENT_TYPES else "public_speech"
    )
    for action, pattern in _ACTION_CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            ledger["action_claims"].append({
                "day": day,
                "speaker": speaker,
                "action": action,
                "target": match.group(1),
                "authority": "player_claim",
                "support_kind": support_kind,
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


def _add_confirmed_action(
    ledger: dict[str, list[dict[str, Any]]],
    event: GameEvent,
) -> None:
    ledger["confirmed_actions"].append({
        "day": event.payload.get("day_number", 0),
        "actor": event.payload.get("actor_id"),
        "action": "hunter_shot",
        "target": event.payload.get("target_id"),
        "authority": "engine",
        "support_kind": "executed_action",
        "source_event": event.type,
    })


def _add_claim_conflicts(
    ledger: dict[str, list[dict[str, Any]]],
) -> None:
    for claim in ledger["action_claims"]:
        for action in ledger["confirmed_actions"]:
            if not _is_related_action_evidence(claim, action):
                continue
            if _action_evidence_matches(claim, action):
                continue
            ledger["claim_conflicts"].append({
                "day": claim["day"],
                "speaker": claim["speaker"],
                "claimed_action": claim["action"],
                "claimed_target": claim["target"],
                "engine_actor": action["actor"],
                "engine_action": action["action"],
                "engine_target": action["target"],
                "authority": "engine",
                "support_kind": "claim_conflict",
                "source_event": action["source_event"],
            })


def _all_engine_action_evidence(game_state: GameState) -> list[dict[str, Any]]:
    evidence = list(build_public_ledger(game_state)["confirmed_actions"])
    witch_ids = [
        player_id
        for player_id, player in game_state.players.items()
        if player.role == "witch"
    ]
    inferred_witch_id = witch_ids[0] if len(witch_ids) == 1 else None
    for event in game_state.events:
        payload = event.payload or {}
        if event.type == "hunter_shot_selected":
            evidence.append({
                "day": payload.get("day_number", 0),
                "actor": payload.get("actor_id") or payload.get("hunter_id"),
                "action": "hunter_shot",
                "target": payload.get("target_id"),
                "source_event": event.type,
            })
        elif event.type == "witch_antidote_used":
            evidence.append({
                "day": payload.get("day_number", 0),
                "actor": (
                    payload.get("actor_id")
                    or payload.get("witch_id")
                    or inferred_witch_id
                ),
                "action": "witch_antidote",
                "target": payload.get("target_id"),
                "source_event": event.type,
            })
    return evidence


def _claim_action_status(
    claim: dict[str, Any],
    engine_actions: list[dict[str, Any]],
) -> str:
    if any(_action_evidence_matches(claim, action) for action in engine_actions):
        return "confirmed"
    if any(_is_related_action_evidence(claim, action) for action in engine_actions):
        return "conflicts_with_engine"
    return "unconfirmed"


def _action_evidence_matches(
    claim: dict[str, Any],
    action: dict[str, Any],
) -> bool:
    actor = action.get("actor")
    return (
        _action_days_are_compatible(claim, action)
        and (actor is None or actor == claim.get("speaker"))
        and action.get("action") == claim.get("action")
        and action.get("target") == claim.get("target")
    )


def _is_related_action_evidence(
    claim: dict[str, Any],
    action: dict[str, Any],
) -> bool:
    actor = action.get("actor")
    return (
        _action_days_are_compatible(claim, action)
        and (
            action.get("action") == claim.get("action")
            or (actor is not None and actor == claim.get("speaker"))
        )
    )


def _action_days_are_compatible(
    claim: dict[str, Any],
    action: dict[str, Any],
) -> bool:
    """显式日次必须一致；零值或缺失日次保留旧事件兼容性。"""
    claim_day = claim.get("day", 0)
    action_day = action.get("day", 0)
    return not claim_day or not action_day or claim_day == action_day
