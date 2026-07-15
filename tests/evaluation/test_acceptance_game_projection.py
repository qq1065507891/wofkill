# -*- coding: utf-8 -*-
"""
验证验收指标只消费完整、可离线复算的游戏投影。

作者: Project contributors
创建日期: 2026-07-15
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_finished_projection_requires_winner_but_aborted_does_not() -> None:
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    common = {
        "game_id": "g-status-contract",
        "players": {"p01": {"role": "villager"}},
        "events": [],
    }
    finished = project_acceptance_game({**common, "status": "finished"})
    aborted = project_acceptance_game({**common, "status": "aborted"})

    assert finished.supported is False
    assert finished.unsupported_reason == "finished_without_winner"
    assert aborted.supported is True
    assert aborted.winning_faction is None


def test_projection_metadata_is_deep_snapshot_and_json_safe() -> None:
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    nested = {"parts": ["before"]}
    source = {
        "game_id": "g-metadata",
        "players": {"p01": {"role": "villager"}},
        "events": [],
        "status": "running",
        "hybrid_result": nested,
        "__source_path": Path("legacy/game.json"),
    }
    projection = project_acceptance_game(source)
    nested["parts"].append("after")

    assert projection.metadata["hybrid_result"] == {"parts": ["before"]}
    assert projection.metadata["__source_path"] == str(Path("legacy/game.json"))
    json.dumps(projection.to_mapping())


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
    assert path.name == "game_g-projection.json"
    assert "speech_fill_rate" not in saved["quality_score"]


def test_game_state_status_fields_are_json_safe_and_default_running() -> None:
    state = GameState(game_id="g-running")

    assert state.status == "running"
    assert state.termination_reason is None


def test_quality_speech_metrics_use_independent_observed_denominators() -> None:
    from scripts.run_real_game import compute_game_quality_score
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    state = GameState(
        game_id="g-speech",
        players={"p01": PlayerState(id="p01", role="villager")},
        events=[
            GameEvent(type="speech", payload={"speaker": "p01", "text": "有内容"}),
            GameEvent(type="speech", payload={"speaker": "p01"}),
            GameEvent(type="action_trace_audit", payload={
                "task_type": "speech",
                "action_trace": {"generated_by": "model", "decision_outcome": "direct_success",
                                 "semantic_repair_audit": {"success": True}},
            }),
            GameEvent(type="action_trace_audit", payload={
                "task_type": "speech",
                "action_trace": {"generated_by": "terminal_fallback",
                                 "decision_outcome": "terminal_fallback",
                                 "semantic_repair_audit": {}},
            }),
        ],
    )

    quality = compute_game_quality_score(project_acceptance_game(state))

    assert quality["speech_non_empty_metrics_supported"] is True
    assert quality["speech_non_empty_observed_count"] == 1
    assert quality["speech_non_empty_rate"] == 1.0
    assert quality["speech_model_success_metrics_supported"] is True
    assert quality["speech_model_success_observed_count"] == 2
    assert quality["speech_model_success_rate"] == 0.5
    assert quality["speech_terminal_fallback_metrics_supported"] is True
    assert quality["speech_terminal_fallback_observed_count"] == 2
    assert quality["speech_terminal_fallback_rate"] == 0.5
    assert quality["speech_semantic_acceptance_metrics_supported"] is True
    assert quality["speech_semantic_acceptance_observed_count"] == 1
    assert quality["speech_semantic_acceptance_rate"] == 1.0
    assert "speech_fill_rate" not in quality


def test_legacy_speech_fields_are_unsupported_but_alias_is_readable() -> None:
    from scripts.run_real_game import compute_game_quality_score, normalize_quality_score
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    state = GameState(
        game_id="g-legacy-speech",
        players={"p01": PlayerState(id="p01", role="villager")},
        events=[
            GameEvent(type="speech", payload={"speaker": "p01"}),
            GameEvent(type="action_trace_audit", payload={
                "task_type": "speech", "action_trace": {},
            }),
        ],
    )
    quality = compute_game_quality_score(project_acceptance_game(state))

    assert quality["speech_non_empty_metrics_supported"] is False
    assert quality["speech_non_empty_rate"] is None
    assert quality["speech_model_success_metrics_supported"] is False
    assert quality["speech_model_success_rate"] is None
    assert quality["speech_terminal_fallback_metrics_supported"] is False
    assert quality["speech_terminal_fallback_rate"] is None
    assert quality["speech_semantic_acceptance_metrics_supported"] is False
    assert quality["speech_semantic_acceptance_rate"] is None
    assert normalize_quality_score({"speech_fill_rate": 0.25})["speech_non_empty_rate"] == 0.25


def test_terminal_semantic_rejects_incomplete_projection() -> None:
    from werewolf_agent.evaluation.acceptance_terminal_semantic_metrics import (
        compute_terminal_semantic_acceptance_metrics,
    )

    identity = {
        "trace_id": "trace-semantic",
        "game_id": "g-semantic-legacy",
        "action_index": 1,
        "task_type": "speech",
    }
    semantic = {
        "repairable": True,
        "success": True,
        "target_preserved": True,
        "speaker_attribution_preserved": True,
        "negation_preserved": True,
        "introduced_claim_count": 0,
        "verified_claim_count": 0,
        "retained_verified_claim_count": 0,
        "fallback_kind": "no_fallback",
    }
    metrics = compute_terminal_semantic_acceptance_metrics([{
        "game_id": "g-semantic-legacy",
        "events": [
            {"type": "semantic_repair_audit", "payload": {**identity, **semantic}},
            {"type": "action_trace_audit", "payload": {
                **identity, "action_trace": {"semantic_repair_audit": semantic},
            }},
        ],
    }])

    assert metrics["semantic_repair_metrics_supported"] is False
    assert metrics["semantic_repair_success_rate"] is None
    assert metrics["semantic_repair_verified_claim_retention_metrics_supported"] is False


@pytest.mark.parametrize(
    (
        "trace", "model_supported", "model_rate", "model_reason",
        "terminal_supported", "terminal_rate", "terminal_reason",
    ),
    [
        ({"generated_by": "model"}, True, 1.0, None,
         False, None, "missing_decision_outcome"),
        ({"decision_outcome": "terminal_fallback"}, False, None, "missing_generated_by",
         True, 1.0, None),
        ({"generated_by": "terminal_fallback"}, True, 0.0, None,
         False, None, "missing_decision_outcome"),
        ({"decision_outcome": "direct_success"}, False, None, "missing_generated_by",
         True, 0.0, None),
        ({"generated_by": "invalid", "decision_outcome": "direct_success"},
         False, None, "invalid_generated_by", True, 0.0, None),
        ({"generated_by": "model", "decision_outcome": "invalid"},
         True, 1.0, None, False, None, "invalid_decision_outcome"),
    ],
)
def test_speech_outcome_metrics_have_independent_closed_observability(
    trace,
    model_supported,
    model_rate,
    model_reason,
    terminal_supported,
    terminal_rate,
    terminal_reason,
) -> None:
    from scripts.run_real_game import compute_game_quality_score
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    state = GameState(
        game_id="g-speech-observability",
        players={"p01": PlayerState(id="p01", role="villager")},
        events=[GameEvent(type="action_trace_audit", payload={
            "task_type": "speech", "action_trace": trace,
        })],
    )
    quality = compute_game_quality_score(project_acceptance_game(state))

    assert quality["speech_model_success_metrics_supported"] is model_supported
    assert quality["speech_model_success_rate"] == model_rate
    assert quality["speech_model_success_unsupported_reason"] == model_reason
    assert quality["speech_terminal_fallback_metrics_supported"] is terminal_supported
    assert quality["speech_terminal_fallback_rate"] == terminal_rate
    assert quality["speech_terminal_fallback_unsupported_reason"] == terminal_reason


def test_load_game_logs_normalizes_legacy_quality_without_rewriting(tmp_path) -> None:
    from werewolf_agent.evaluation.balance_audit import load_game_logs

    legacy_path = tmp_path / "legacy.json"
    conflict_path = tmp_path / "conflict.json"
    legacy_path.write_text(json.dumps({
        "game_id": "legacy",
        "quality_score": {"speech_fill_rate": 0.25},
    }), encoding="utf-8")
    conflict_path.write_text(json.dumps({
        "game_id": "conflict",
        "quality_score": {
            "speech_fill_rate": 0.25,
            "speech_non_empty_rate": 0.75,
        },
    }), encoding="utf-8")
    before = [path.read_text(encoding="utf-8") for path in (legacy_path, conflict_path)]

    legacy, conflict = load_game_logs([legacy_path, conflict_path])

    assert legacy["quality_score"]["speech_non_empty_rate"] == 0.25
    assert conflict["quality_score"]["speech_non_empty_rate"] == 0.75
    assert [path.read_text(encoding="utf-8") for path in (legacy_path, conflict_path)] == before
