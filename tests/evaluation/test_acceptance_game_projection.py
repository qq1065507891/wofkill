# -*- coding: utf-8 -*-
"""
验证验收指标只消费完整、可离线复算的游戏投影。

作者: Project contributors
创建日期: 2026-07-15
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

from werewolf_agent.core.models import GameEvent, GameState, PlayerState


def _completed_state() -> GameState:
    players = {
        "p01": PlayerState(id="p01", role="seer"),
        "p02": PlayerState(id="p02", role="werewolf"),
    }
    return GameState(
        game_id="g-projection",
        players=players,
        winning_faction="good",
        events=[
            GameEvent(type="victory", payload={"winner": "good"}),
            GameEvent(type="reflection_complete", payload={
                "player_count": 2,
                "entries": [
                    {"player_id": player_id, "decision_id": f"reflection:g-projection:{player_id}",
                     "verification": {"decision_id": f"reflection:g-projection:{player_id}",
                                      "verified_lessons": [], "rejected_fact_count": 0,
                                      "rejected_lesson_count": 0}}
                    for player_id in players
                ],
            }),
            GameEvent(type="reflection_persistence_audit", payload={
                "expected_entry_count": 0,
                "entries": [],
                "persistence_complete": True,
                "rollback_complete": True,
            }),
        ],
    )


def test_completed_state_projects_players_winner_and_finished_status() -> None:
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    projection = project_acceptance_game(_completed_state(), steps=7)

    assert projection.game_id == "g-projection"
    assert set(projection.players) == {"p01", "p02"}
    assert projection.winning_faction == "good"
    assert projection.status == "finished"
    assert projection.steps == 7


def test_projection_is_snapshot_and_explicit_aborted_status_is_authoritative() -> None:
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    source = {
        "game_id": "g-aborted",
        "status": "aborted",
        "winning_faction": "good",
        "players": {"p01": {"role": "villager", "facts": ["before"]}},
        "events": [{"type": "audit", "payload": {"nested": ["before"]}}],
    }
    projection = project_acceptance_game(source)
    source["players"]["p01"]["facts"].append("after")
    source["events"][0]["payload"]["nested"].append("after")

    assert projection.status == "aborted"
    assert projection.players["p01"]["facts"] == ["before"]
    assert projection.events[0]["payload"]["nested"] == ["before"]


def test_legacy_projection_infers_status_but_does_not_fake_missing_players() -> None:
    from werewolf_agent.evaluation.acceptance_audit import compute_acceptance_audit_metrics
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    completed = project_acceptance_game({
        "game_id": "legacy-finished",
        "winning_faction": "werewolf",
        "events": [],
    })
    running = project_acceptance_game({"game_id": "legacy-running", "events": []})
    metrics = compute_acceptance_audit_metrics([completed])

    assert completed.status == "finished"
    assert running.status == "running"
    assert completed.supported is False
    assert completed.unsupported_reason == "missing_players"
    assert metrics["acceptance_projection_supported"] is False
    assert metrics["acceptance_projection_unsupported_reason"] == "missing_players"
    assert metrics["possible_world_metrics_supported"] is False
    assert metrics["possible_world_unique_rate"] is None


def test_incomplete_projection_nulls_rates_even_when_metric_events_exist() -> None:
    from werewolf_agent.evaluation.acceptance_audit import compute_acceptance_audit_metrics

    metrics = compute_acceptance_audit_metrics([{
        "game_id": "legacy-events-only",
        "events": [{
            "type": "action_trace_audit",
            "payload": {"action_trace": {"world_model_audit": {
                "possible_worlds": {"top_worlds": [{
                    "label": "legacy-world",
                    "key_assignments": {"p01": "werewolf"},
                    "why": [],
                }]},
                "authoritative_world_identities": [],
                "public_evidence_ids": [],
            }}},
        }],
    }])

    assert metrics["acceptance_projection_unsupported_reason"] == "missing_players"
    assert metrics["possible_world_metrics_supported"] is False
    assert metrics["possible_world_unique_rate"] is None
    assert metrics["possible_world_evidence_coverage_rate"] is None


def test_saved_projection_recomputes_identical_final_quality(tmp_path) -> None:
    from scripts.run_real_game import compute_game_quality_score, save_game_log
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    runner = SimpleNamespace(
        game_id="g-projection",
        state=_completed_state(),
        step_count=7,
    )
    projection = project_acceptance_game(runner.state, steps=runner.step_count)
    final_quality = compute_game_quality_score(projection)
    runner.state = replace(
        runner.state,
        players={"mutated": PlayerState(id="mutated", role="werewolf")},
        events=[GameEvent(type="mutated_after_projection", payload={})],
    )

    path = save_game_log(
        runner,
        elapsed=1.0,
        output_dir=tmp_path,
        projection=projection,
        quality_score=final_quality,
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    offline_quality = compute_game_quality_score(project_acceptance_game(saved))

    assert saved["status"] == "finished"
    assert set(saved["players"]) == {"p01", "p02"}
    assert saved["events"][-1]["type"] == "reflection_persistence_audit"
    assert saved["quality_score"] == final_quality
    assert offline_quality == final_quality


def test_game_state_status_fields_are_json_safe_and_default_running() -> None:
    state = GameState(game_id="g-running")

    assert state.status == "running"
    assert state.termination_reason is None


def test_quality_uses_distinct_speech_rates_and_read_only_legacy_alias() -> None:
    from scripts.run_real_game import compute_game_quality_score
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    state = GameState(
        game_id="g-speech",
        players={"p01": PlayerState(id="p01", role="villager")},
        events=[
            GameEvent(type="speech", payload={"speaker": "p01", "text": "有内容"}),
            GameEvent(type="speech", payload={"speaker": "p01", "text": ""}),
            GameEvent(type="action_trace_audit", payload={
                "task_type": "speech",
                "action_trace": {"generated_by": "model", "decision_outcome": "direct_success",
                                 "semantic_repair_audit": {"success": True}},
            }),
            GameEvent(type="action_trace_audit", payload={
                "task_type": "speech",
                "action_trace": {"generated_by": "terminal_fallback",
                                 "decision_outcome": "terminal_fallback",
                                 "semantic_repair_audit": {"success": False}},
            }),
        ],
    )

    quality = compute_game_quality_score(project_acceptance_game(state))

    assert quality["speech_non_empty_rate"] == 0.5
    assert quality["speech_model_success_rate"] == 0.5
    assert quality["speech_terminal_fallback_rate"] == 0.5
    assert quality["speech_semantic_acceptance_rate"] == 0.5
    assert quality["speech_fill_rate"] == quality["speech_non_empty_rate"]
