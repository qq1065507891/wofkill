# -*- coding: utf-8 -*-
"""
投影赛后反思验证、持久化与污染防护验收指标。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-18
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from werewolf_agent.evaluation.acceptance_shared import (
    _is_non_negative_int,
    _non_negative_int,
)
from werewolf_agent.evaluation.game_projection import (
    normalize_acceptance_games,
    projection_support,
)


def compute_reflection_acceptance_metrics(
    games: Iterable[Any],
) -> dict[str, Any]:
    """扫描每局最新反思事务并投影持久化验收指标。"""
    return _compute_reflection_acceptance_metrics_from_normalized(
        normalize_acceptance_games(games)
    )


def _compute_reflection_acceptance_metrics_from_normalized(
    games: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """消费同一调用链刚完成验证的不可变游戏快照。"""
    projection_is_supported, projection_reason = projection_support(games)
    rejected_facts = 0
    rejected_lessons = 0
    reflection_contamination_source_complete = True
    persisted_rejected_facts = 0
    reflection_completed_game_count = 0
    reflection_audited_game_count = 0
    transaction_failure_reason: str | None = None

    for game in games:
        latest_reflection_index = -1
        latest_reflection_payload: Mapping[str, Any] | None = None
        persistence_events: list[tuple[int, dict[str, Any]]] = []
        for event_index, event in enumerate(game.get("events", [])):
            if not isinstance(event, Mapping):
                continue
            event_type = event.get("type")
            payload = event.get("payload") or {}
            if event_type == "reflection_complete" and isinstance(payload, Mapping):
                latest_reflection_index = event_index
                latest_reflection_payload = payload
            if event_type == "reflection_persistence_audit" and isinstance(
                payload, Mapping
            ):
                persistence_events.append((event_index, payload))

        for entry in _reflection_entries(latest_reflection_payload):
            verification = entry.get("verification")
            if not isinstance(verification, Mapping):
                continue
            rejected_facts += _non_negative_int(
                verification.get("rejected_fact_count")
            )
            rejected_lessons += _non_negative_int(
                verification.get("rejected_lesson_count")
            )

        winning_faction = game.get("winning_faction")
        is_completed_game = (
            isinstance(winning_faction, str)
            and winning_faction in {"good", "werewolf"}
        )
        if is_completed_game:
            reflection_completed_game_count += 1
            player_ids = set(game.get("players", {})) if isinstance(
                game.get("players"), Mapping
            ) else set()
            post_reflection_transactions = [
                item for item in persistence_events
                if item[0] > latest_reflection_index
            ]
            game_id = game.get("game_id")
            audited, rejected_count, reason = _audit_reflection_transaction(
                game_id=game_id if isinstance(game_id, str) else "",
                player_ids=player_ids,
                reflection_payload=latest_reflection_payload,
                persistence_events=post_reflection_transactions,
            )
            if audited:
                reflection_audited_game_count += 1
                persisted_rejected_facts += rejected_count
            else:
                reflection_contamination_source_complete = False
                transaction_failure_reason = transaction_failure_reason or reason

    supported = (
        projection_is_supported
        and reflection_completed_game_count > 0
        and reflection_audited_game_count == reflection_completed_game_count
        and reflection_contamination_source_complete
    )
    return {
        "reflection_rejected_fact_count": rejected_facts,
        "reflection_rejected_lesson_count": rejected_lessons,
        "reflection_completed_game_count": reflection_completed_game_count,
        "reflection_audited_game_count": reflection_audited_game_count,
        "reflection_contamination_metrics_supported": supported,
        "reflection_contamination_metrics_unsupported_reason": (
            projection_reason if not projection_is_supported
            else None if supported
            else transaction_failure_reason or "incomplete_reflection_audit"
        ),
        "reflection_persisted_rejected_fact_count": (
            persisted_rejected_facts if supported else None
        ),
    }


def _reflection_entries(
    payload: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    """只返回结构有效的局级反思玩家条目。"""
    if not isinstance(payload, Mapping):
        return []
    entries = payload.get("entries")
    if not isinstance(entries, (list, tuple)):
        return []
    return [entry for entry in entries if isinstance(entry, Mapping)]


def _audit_reflection_transaction(
    *,
    game_id: str,
    player_ids: set[str],
    reflection_payload: Mapping[str, Any] | None,
    persistence_events: list[tuple[int, dict[str, Any]]],
) -> tuple[bool, int, str]:
    """验证 complete/partial 事务及 decision/claim/entry 全链。"""
    if not _reflection_payload_matches_players(reflection_payload, player_ids):
        return False, 0, "incomplete_reflection_audit"
    assert reflection_payload is not None
    reflection_status = reflection_payload.get("status")
    if reflection_status == "no_valid_entries":
        return False, 0, "reflection_no_valid_entries"
    if reflection_status == "persistence_failed":
        return False, 0, "reflection_persistence_failed"
    if (
        not isinstance(reflection_status, str)
        or reflection_status not in {"complete", "partial"}
    ):
        return False, 0, "incomplete_reflection_audit"
    if len(persistence_events) != 1:
        return False, 0, "incomplete_reflection_audit"
    persistence = persistence_events[0][1]
    status = persistence.get("status")
    if status == "no_valid_entries":
        return False, 0, "reflection_no_valid_entries"
    if status == "persistence_failed":
        return False, 0, "reflection_persistence_failed"
    if (
        not isinstance(status, str)
        or status not in {"complete", "partial"}
        or status != reflection_status
    ):
        return False, 0, "incomplete_reflection_audit"

    rows = persistence.get("entries")
    if (
        not isinstance(rows, (list, tuple))
        or not rows
        or not _matches_count(
            persistence.get("expected_entry_count"), len(rows)
        )
        or persistence.get("persistence_complete") is not True
        or persistence.get("rollback_complete") is not True
    ):
        return False, 0, "incomplete_reflection_audit"
    event_entries = {
        entry.get("player_id"): entry
        for entry in _reflection_entries(reflection_payload)
        if isinstance(entry.get("player_id"), str)
    }
    audit_entries = {
        row.get("player_id"): row
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("player_id"), str)
        and row.get("player_id")
    }
    if len(audit_entries) != len(rows):
        return False, 0, "incomplete_reflection_audit"

    eligible_players = {
        player_id for player_id, entry in event_entries.items()
        if _entry_has_verified_lessons(entry)
    }
    if not eligible_players or set(audit_entries) != eligible_players:
        return False, 0, "incomplete_reflection_audit"
    failed_players = player_ids - eligible_players
    explicit_failure_count = sum(
        1 for player_id in failed_players
        if _entry_has_explicit_failure(
            event_entries[player_id],
            expected_decision_id=_canonical_decision_id(game_id, player_id),
        )
    )
    valid_entry_count = reflection_payload.get("valid_entry_count")
    failure_count = reflection_payload.get("failure_count")
    if (
        not _matches_count(valid_entry_count, len(eligible_players))
        or not _matches_count(failure_count, explicit_failure_count)
    ):
        return False, 0, "incomplete_reflection_audit"
    if status == "complete" and (eligible_players != player_ids or failed_players):
        return False, 0, "incomplete_reflection_audit"
    if status == "partial" and (
        not failed_players
        or not all(
            _entry_has_explicit_failure(
                event_entries[player_id],
                expected_decision_id=_canonical_decision_id(game_id, player_id),
            )
            for player_id in failed_players
        )
    ):
        return False, 0, "incomplete_reflection_audit"

    entry_ids: set[str] = set()
    for player_id in eligible_players:
        event_entry = event_entries[player_id]
        verification = event_entry.get("verification")
        audit_entry = audit_entries[player_id]
        if not isinstance(verification, Mapping):
            return False, 0, "incomplete_reflection_audit"
        decision_id = verification.get("decision_id")
        event_decision_id = event_entry.get("decision_id")
        verified_claim_ids = verification.get("verified_claim_ids", [])
        audited_claim_ids = audit_entry.get("verified_claim_ids")
        entry_id = audit_entry.get("entry_id")
        transaction_state = event_entry.get("transaction_state")
        canonical_decision_id = _canonical_decision_id(game_id, player_id)
        if (
            not isinstance(decision_id, str)
            or not decision_id
            or decision_id != canonical_decision_id
            or event_decision_id != decision_id
            or audit_entry.get("decision_id") != decision_id
            or not verified_claim_ids
            or not _same_unique_identifiers(verified_claim_ids, audited_claim_ids)
            or not isinstance(transaction_state, str)
            or transaction_state not in {
                "lessons_verified", "persisted",
            }
            or (
                transaction_state == "persisted"
                and event_entry.get("entry_id") != entry_id
            )
            or (
                transaction_state == "lessons_verified"
                and event_entry.get("entry_id") is not None
                and event_entry.get("entry_id") != ""
            )
            or not isinstance(entry_id, str)
            or entry_id != f"reflection_{game_id}_{player_id}"
            or entry_id in entry_ids
            or audit_entry.get("row_found") is not True
            or audit_entry.get("persistence_complete") is not True
            or not _matches_count(
                audit_entry.get("persisted_rejected_fact_count"), 0
            )
        ):
            return False, 0, "incomplete_reflection_audit"
        entry_ids.add(entry_id)
    return True, 0, ""


def _entry_has_verified_lessons(entry: Mapping[str, Any]) -> bool:
    verification = entry.get("verification")
    lessons = verification.get("verified_lessons") if isinstance(
        verification, Mapping
    ) else None
    return isinstance(lessons, (list, tuple)) and any(
        isinstance(lesson, Mapping)
        and isinstance(lesson.get("lesson_id"), str)
        and bool(lesson.get("lesson_id"))
        and isinstance(lesson.get("abstraction"), str)
        and bool(lesson.get("abstraction", "").strip())
        for lesson in lessons
    )


def _entry_has_explicit_failure(
    entry: Mapping[str, Any],
    *,
    expected_decision_id: str,
) -> bool:
    transaction_state = entry.get("transaction_state")
    expected_failure_stage = {
        "not_requested": "generated",
        "generated": "schema_validated",
        "schema_validated": "facts_verified",
        "facts_verified": "lessons_verified",
    }.get(transaction_state) if isinstance(transaction_state, str) else None
    verification = entry.get("verification")
    decision_id = entry.get("decision_id")
    return (
        expected_failure_stage is not None
        and entry.get("failure_stage") == expected_failure_stage
        and isinstance(entry.get("failure_stage"), str)
        and bool(entry.get("failure_stage"))
        and isinstance(entry.get("failure_code"), str)
        and bool(entry.get("failure_code"))
        and isinstance(decision_id, str)
        and bool(decision_id)
        and decision_id == expected_decision_id
        and isinstance(verification, Mapping)
        and verification.get("decision_id") == decision_id
    )


def _same_unique_identifiers(left: Any, right: Any) -> bool:
    """身份列表必须都是无重复字符串，且顺序无关地精确相等。"""
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    if any(not isinstance(item, str) or not item for item in (*left, *right)):
        return False
    return (
        len(left) == len(set(left))
        and len(right) == len(set(right))
        and set(left) == set(right)
    )


def _matches_count(value: Any, expected: int) -> bool:
    """只接受非布尔、非负的原生整数，并与权威计数精确相等。"""
    return _is_non_negative_int(value) and value == expected


def _canonical_decision_id(game_id: str, player_id: str) -> str:
    """反思 decision 只允许绑定当前游戏和当前玩家。"""
    return f"reflection:{game_id}:{player_id}"


def _reflection_payload_matches_players(
    payload: dict[str, Any] | None,
    player_ids: set[str],
) -> bool:
    """完成局必须逐玩家反思；只有真实零玩家局可接受零条目。"""
    if not isinstance(payload, Mapping):
        return False
    entries = payload.get("entries")
    if (
        not isinstance(entries, (list, tuple))
        or not _matches_count(payload.get("player_count"), len(player_ids))
    ):
        return False
    entry_players = [
        entry.get("player_id")
        for entry in entries
        if isinstance(entry, Mapping) and isinstance(entry.get("player_id"), str)
    ]
    return (
        len(entry_players) == len(entries)
        and len(entry_players) == len(set(entry_players))
        and set(entry_players) == player_ids
    )
