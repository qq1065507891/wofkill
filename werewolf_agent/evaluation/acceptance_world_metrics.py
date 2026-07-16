# -*- coding: utf-8 -*-
"""
投影可能世界唯一性与公开证据覆盖验收指标。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-16
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.evaluation.acceptance_shared import _game_player_roles
from werewolf_agent.evaluation.game_projection import (
    ensure_normalized_acceptance_games,
    projection_support,
)
from werewolf_agent.evaluation.world_evidence_audit import (
    _current_game_public_evidence_refs,
    _public_assignment_support_before,
    _support_matches_world,
    _valid_unique_world_count,
)


def compute_world_acceptance_metrics(
    games: list[dict[str, Any]],
) -> dict[str, Any]:
    """扫描世界模型审计并投影其验收指标。"""
    games = ensure_normalized_acceptance_games(games)
    projection_is_supported, projection_reason = projection_support(games)
    world_groups: list[
        tuple[
            list[dict[str, Any]],
            dict[str, set[tuple[str, str]]],
            list[dict[str, Any]],
            dict[str, str] | None,
        ]
    ] = []
    for game in games:
        for event_index, event in enumerate(game.get("events", [])):
            if not isinstance(event, dict) or event.get("type") != "action_trace_audit":
                continue
            payload = event.get("payload") or {}
            trace = payload.get("action_trace")
            if not isinstance(trace, dict):
                continue
            audit = trace.get("world_model_audit")
            possible = audit.get("possible_worlds") if isinstance(audit, dict) else None
            top_worlds = possible.get("top_worlds") if isinstance(possible, dict) else None
            if not isinstance(top_worlds, list):
                continue
            authoritative_refs = audit.get("public_evidence_ids")
            authoritative = _current_game_public_evidence_refs(
                game,
                authoritative_refs if isinstance(authoritative_refs, list) else [],
            )
            semantic_support = _public_assignment_support_before(game, event_index)
            valid_support = {
                ref: concepts for ref, concepts in semantic_support.items()
                if ref in authoritative
            }
            identity_proofs = audit.get("authoritative_world_identities")
            world_groups.append((
                [item for item in top_worlds if isinstance(item, dict)],
                valid_support,
                identity_proofs if isinstance(identity_proofs, list) else [],
                _game_player_roles(game),
            ))

    world_count = sum(len(group) for group, _, _, _ in world_groups)
    unique_world_count = sum(
        _valid_unique_world_count(group, proofs, expected_roles)
        for group, _, proofs, expected_roles in world_groups
    )
    evidence_covered = sum(
        isinstance(world.get("why"), list)
        and bool(world["why"])
        and all(
            isinstance(ref, str)
            and ref in valid_support
            and _support_matches_world(valid_support[ref], world)
            for ref in world["why"]
        )
        for group, valid_support, _, _ in world_groups
        for world in group
    )
    return {
        "possible_world_metrics_supported": projection_is_supported and world_count > 0,
        "possible_world_metrics_unsupported_reason": (
            projection_reason if not projection_is_supported
            else None if world_count else "no_possible_worlds"
        ),
        "possible_world_prompt_count": len(world_groups),
        "possible_world_total_count": world_count,
        "possible_world_unique_count": unique_world_count,
        "possible_world_unique_rate": (
            unique_world_count / world_count
            if projection_is_supported and world_count else None
        ),
        "possible_world_evidence_covered_count": evidence_covered,
        "possible_world_evidence_coverage_rate": (
            evidence_covered / world_count
            if projection_is_supported and world_count else None
        ),
    }
