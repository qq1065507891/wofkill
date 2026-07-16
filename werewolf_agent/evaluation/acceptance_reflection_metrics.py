# -*- coding: utf-8 -*-
"""
投影赛后反思验证、持久化与污染防护验收指标。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-16
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.evaluation.acceptance_shared import (
    _is_non_negative_int,
    _non_negative_int,
)
from werewolf_agent.evaluation.game_projection import (
    normalize_acceptance_games,
    projection_support,
)


def compute_reflection_acceptance_metrics(
    games: list[dict[str, Any]],
) -> dict[str, Any]:
    """扫描每局最新反思事务并投影持久化验收指标。"""
    games = normalize_acceptance_games(games)
    projection_is_supported, projection_reason = projection_support(games)
    rejected_facts = 0
    rejected_lessons = 0
    reflection_persistence_entry_count = 0
    reflection_contamination_source_complete = True
    persisted_rejected_facts = 0
    reflection_completed_game_count = 0
    reflection_audited_game_count = 0

    for game in games:
        latest_reflections: dict[str, tuple[int, dict[str, Any]]] = {}
        latest_persistence: dict[str, tuple[int, dict[str, Any], bool]] = {}
        reflection_complete_seen = False
        latest_reflection_index = -1
        latest_reflection_payload: dict[str, Any] | None = None
        persistence_events: list[tuple[int, dict[str, Any]]] = []
        for event_index, event in enumerate(game.get("events", [])):
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            payload = event.get("payload") or {}
            if event_type == "reflection_complete":
                reflection_complete_seen = True
                latest_reflection_index = event_index
                latest_reflection_payload = payload
                latest_reflections = {}
                for entry in payload.get("entries", []):
                    if not isinstance(entry, dict):
                        continue
                    player_id = entry.get("player_id")
                    verification = entry.get("verification")
                    if isinstance(player_id, str) and isinstance(verification, dict):
                        latest_reflections[player_id] = (event_index, verification)
            if event_type == "reflection_persistence_audit":
                persistence_events.append((event_index, payload))

        post_reflection_transactions = [
            item for item in persistence_events if item[0] > latest_reflection_index
        ]
        authoritative_persistence: dict[str, Any] | None = None
        transaction_structure_valid = len(post_reflection_transactions) == 1
        if transaction_structure_valid:
            persistence_index, authoritative_persistence = post_reflection_transactions[0]
            entries = authoritative_persistence.get("entries")
            transaction_structure_valid = (
                isinstance(entries, list)
                and authoritative_persistence.get("expected_entry_count") == len(entries)
                and authoritative_persistence.get("persistence_complete") is True
                and authoritative_persistence.get("rollback_complete") is True
            )
            if transaction_structure_valid and isinstance(entries, list):
                player_ids = [
                    entry.get("player_id") for entry in entries
                    if isinstance(entry, dict)
                    and isinstance(entry.get("player_id"), str)
                    and entry.get("player_id")
                ]
                entry_ids = [
                    entry.get("entry_id") for entry in entries
                    if isinstance(entry, dict)
                    and isinstance(entry.get("entry_id"), str)
                    and entry.get("entry_id")
                ]
                transaction_structure_valid = (
                    len(player_ids) == len(entries) == len(entry_ids)
                    and len(player_ids) == len(set(player_ids))
                    and len(entry_ids) == len(set(entry_ids))
                )
                if transaction_structure_valid:
                    latest_persistence = {
                        entry["player_id"]: (persistence_index, entry, True)
                        for entry in entries
                    }

        for _, verification in latest_reflections.values():
            rejected_facts += _non_negative_int(verification.get("rejected_fact_count"))
            rejected_lessons += _non_negative_int(
                verification.get("rejected_lesson_count")
            )
        eligible_reflection_players: set[str] = set()
        game_reflection_complete = True
        for player_id, (reflection_index, verification) in latest_reflections.items():
            lessons = verification.get("verified_lessons")
            expects_persistence = isinstance(lessons, list) and any(
                isinstance(lesson, dict)
                and isinstance(lesson.get("abstraction"), str)
                and bool(lesson["abstraction"].strip())
                for lesson in lessons
            )
            if not expects_persistence:
                continue
            eligible_reflection_players.add(player_id)
            persistence = latest_persistence.get(player_id)
            expected_decision_id = verification.get("decision_id")
            if (
                persistence is None
                or persistence[0] <= reflection_index
                or persistence[1].get("decision_id") != expected_decision_id
            ):
                reflection_contamination_source_complete = False
                game_reflection_complete = False
        if (
            not transaction_structure_valid
            or set(latest_persistence) != eligible_reflection_players
            or (
                authoritative_persistence is not None
                and authoritative_persistence.get("expected_entry_count")
                != len(eligible_reflection_players)
            )
        ):
            reflection_contamination_source_complete = False
            game_reflection_complete = False
        for _, persistence, audit_complete in latest_persistence.values():
            reflection_persistence_entry_count += 1
            persisted_value = persistence.get("persisted_rejected_fact_count")
            entry_id = persistence.get("entry_id")
            entry_complete = (
                audit_complete
                and isinstance(entry_id, str)
                and bool(entry_id)
                and persistence.get("row_found") is True
                and persistence.get("persistence_complete") is True
            )
            if entry_complete and _is_non_negative_int(persisted_value):
                persisted_rejected_facts += persisted_value
            else:
                reflection_contamination_source_complete = False
                game_reflection_complete = False

        is_completed_game = game.get("winning_faction") in {"good", "werewolf"}
        if is_completed_game:
            reflection_completed_game_count += 1
            player_ids = set(game.get("players", {})) if isinstance(
                game.get("players"), dict
            ) else set()
            reflection_payload_valid = _reflection_payload_matches_players(
                latest_reflection_payload, player_ids
            )
            if (
                not reflection_complete_seen
                or authoritative_persistence is None
                or not transaction_structure_valid
                or not reflection_payload_valid
            ):
                game_reflection_complete = False
            if game_reflection_complete:
                reflection_audited_game_count += 1
            else:
                reflection_contamination_source_complete = False

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
            else None if supported else "incomplete_reflection_audit"
        ),
        "reflection_persisted_rejected_fact_count": (
            persisted_rejected_facts if supported else None
        ),
    }


def _reflection_payload_matches_players(
    payload: dict[str, Any] | None,
    player_ids: set[str],
) -> bool:
    """完成局必须逐玩家反思；只有真实零玩家局可接受零条目。"""
    if not isinstance(payload, dict):
        return False
    entries = payload.get("entries")
    if not isinstance(entries, list) or payload.get("player_count") != len(player_ids):
        return False
    entry_players = [
        entry.get("player_id")
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("player_id"), str)
    ]
    return (
        len(entry_players) == len(entries)
        and len(entry_players) == len(set(entry_players))
        and set(entry_players) == player_ids
    )
