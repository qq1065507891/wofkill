# -*- coding: utf-8 -*-
"""
投影终局后调用与语义修复验收指标。

作者: Project contributors
创建日期: 2026-07-14
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.evaluation.acceptance_shared import (
    _is_non_negative_int,
    _non_negative_int,
)

_SEMANTIC_FALLBACK_KINDS = frozenset({
    "no_fallback",
    "generic_template",
    "target_specific",
    "verified_claim",
    "task_specific",
})


def compute_terminal_semantic_acceptance_metrics(
    games: list[dict[str, Any]],
) -> dict[str, Any]:
    """扫描终局与语义事件并投影对应验收指标。"""
    semantic_rows: list[dict[str, Any]] = []
    semantic_eligible_count = 0
    semantic_reconciliation_complete = True
    post_win_calls = 0

    for game in games:
        victory_seen = False
        semantic_events_by_identity: dict[
            tuple[str, str, int, str], list[dict[str, Any]]
        ] = {}
        semantic_trace_identities: list[tuple[str, str, int, str]] = []
        for event in game.get("events", []):
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            payload = event.get("payload") or {}
            if event_type == "victory":
                victory_seen = True
                continue
            if victory_seen and event_type in {
                "action_trace_audit",
                "model_execution_audit",
                "wolf_team_plan",
            }:
                task = str(payload.get("task_type") or payload.get("phase") or "")
                if task not in {"reflection", "post_game_reflection"}:
                    post_win_calls += 1
            if event_type == "semantic_repair_audit" and payload.get("repairable") is True:
                identity = _semantic_identity(payload)
                if identity is not None:
                    semantic_events_by_identity.setdefault(identity, []).append(payload)
            if event_type != "action_trace_audit":
                continue
            trace = payload.get("action_trace")
            if not isinstance(trace, dict):
                continue
            semantic = trace.get("semantic_repair_audit")
            if isinstance(semantic, dict) and semantic.get("repairable") is True:
                semantic_eligible_count += 1
                identity = _semantic_identity(payload)
                if identity is None:
                    semantic_reconciliation_complete = False
                else:
                    semantic_trace_identities.append(identity)

        for identity in semantic_trace_identities:
            matches = semantic_events_by_identity.get(identity, [])
            if len(matches) == 1:
                semantic_rows.append(matches[0])
            else:
                semantic_reconciliation_complete = False
        if any(
            identity not in semantic_trace_identities or len(rows) != 1
            for identity, rows in semantic_events_by_identity.items()
        ):
            semantic_reconciliation_complete = False

    semantic_count = semantic_eligible_count
    semantic_source_complete = (
        semantic_count > 0
        and semantic_reconciliation_complete
        and len(semantic_rows) == semantic_count
        and all(
            isinstance(row.get("success"), bool)
            and isinstance(row.get("target_preserved"), bool)
            and isinstance(row.get("speaker_attribution_preserved"), bool)
            and isinstance(row.get("negation_preserved"), bool)
            and _is_non_negative_int(row.get("introduced_claim_count"))
            and row.get("fallback_kind") in _SEMANTIC_FALLBACK_KINDS
            and (
                (row.get("success") is True and row.get("fallback_kind") == "no_fallback")
                or (
                    row.get("success") is False
                    and row.get("fallback_kind") != "no_fallback"
                )
            )
            for row in semantic_rows
        )
    )
    semantic_success = sum(
        row.get("success") is True
        and row.get("target_preserved") is True
        and row.get("speaker_attribution_preserved") is True
        and row.get("negation_preserved") is True
        and row.get("introduced_claim_count") == 0
        and row.get("retained_verified_claim_count") == row.get("verified_claim_count")
        for row in semantic_rows
    )
    target_preserved = sum(row.get("target_preserved") is True for row in semantic_rows)
    speaker_attribution_preserved = sum(
        row.get("speaker_attribution_preserved") is True for row in semantic_rows
    )
    negation_preserved = sum(
        row.get("negation_preserved") is True for row in semantic_rows
    )
    no_new_claim = (
        sum(
            _non_negative_int(row.get("introduced_claim_count")) == 0
            for row in semantic_rows
        )
        if semantic_source_complete
        else 0
    )
    retention_source_complete = semantic_count > 0 and all(
        _is_non_negative_int(row.get("verified_claim_count"))
        and _is_non_negative_int(row.get("retained_verified_claim_count"))
        and row["retained_verified_claim_count"] <= row["verified_claim_count"]
        for row in semantic_rows
    )
    retention_rows = (
        [
            row
            for row in semantic_rows
            if _non_negative_int(row.get("verified_claim_count")) > 0
        ]
        if retention_source_complete
        else []
    )
    retained_verified_claims = sum(
        row["retained_verified_claim_count"] == row["verified_claim_count"]
        for row in retention_rows
    )
    generic_template_count = (
        sum(row.get("fallback_kind") == "generic_template" for row in semantic_rows)
        if semantic_source_complete
        else None
    )
    return {
        "terminal_post_win_game_model_call_count": post_win_calls,
        "semantic_repair_metrics_supported": semantic_source_complete,
        "semantic_repair_eligible_count": semantic_count,
        "semantic_repair_success_count": semantic_success,
        "semantic_repair_success_rate": (
            semantic_success / semantic_count if semantic_source_complete else None
        ),
        "semantic_repair_target_preservation_rate": (
            target_preserved / semantic_count if semantic_source_complete else None
        ),
        "semantic_repair_speaker_attribution_preservation_rate": (
            speaker_attribution_preserved / semantic_count
            if semantic_source_complete else None
        ),
        "semantic_repair_negation_preservation_rate": (
            negation_preserved / semantic_count if semantic_source_complete else None
        ),
        "semantic_repair_no_new_claim_rate": (
            no_new_claim / semantic_count if semantic_source_complete else None
        ),
        "semantic_repair_generic_template_count": generic_template_count,
        "semantic_repair_verified_claim_retention_metrics_supported": (
            retention_source_complete and bool(retention_rows)
        ),
        "semantic_repair_verified_claim_retention_eligible_count": len(retention_rows),
        "semantic_repair_verified_claim_retained_count": retained_verified_claims,
        "semantic_repair_verified_claim_retention_rate": (
            retained_verified_claims / len(retention_rows) if retention_rows else None
        ),
    }


def _semantic_identity(
    payload: dict[str, Any],
) -> tuple[str, str, int, str] | None:
    trace_id = payload.get("trace_id")
    game_id = payload.get("game_id")
    action_index = payload.get("action_index")
    task_type = payload.get("task_type")
    if (
        not isinstance(trace_id, str) or not trace_id
        or not isinstance(game_id, str) or not game_id
        or not isinstance(action_index, int) or isinstance(action_index, bool)
        or not isinstance(task_type, str) or not task_type
    ):
        return None
    return trace_id, game_id, action_index, task_type
