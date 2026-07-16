# -*- coding: utf-8 -*-
"""
验证验收指标只消费完整、可离线复算的游戏投影。

作者: Project contributors
创建日期: 2026-07-15
修改日期: 2026-07-16
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
    assert "facts" not in projection.players["p01"]
    assert tuple(projection.events[0]["payload"]["nested"]) == ("before",)


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

    assert tuple(projection.metadata["hybrid_result"]["parts"]) == ("before",)
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


def test_quality_speech_metrics_use_one_complete_opportunity_denominator() -> None:
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

    assert quality["speech_opportunity_count"] == 2
    assert quality["speech_non_empty_metrics_supported"] is False
    assert quality["speech_non_empty_observed_count"] == 1
    assert quality["speech_non_empty_rate"] is None
    assert quality["speech_model_success_metrics_supported"] is True
    assert quality["speech_model_success_observed_count"] == 2
    assert quality["speech_model_success_rate"] == 0.5
    assert quality["speech_terminal_fallback_metrics_supported"] is True
    assert quality["speech_terminal_fallback_observed_count"] == 2
    assert quality["speech_terminal_fallback_rate"] == 0.5
    assert quality["speech_semantic_acceptance_metrics_supported"] is False
    assert quality["speech_semantic_acceptance_observed_count"] == 1
    assert quality["speech_semantic_acceptance_rate"] is None
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


def test_projection_is_recursively_immutable_and_redacts_player_extras() -> None:
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    projection = project_acceptance_game({
        "game_id": "g-private-player",
        "players": {"p01": {
            "id": "p01", "role": "seer", "alive": True, "faction": "good",
            "private_prompt": "SECRET", "nested": {"secret": "SECRET"},
        }},
        "events": [{"type": "audit", "payload": {"items": ["before"]}}],
        "deaths": [],
    })

    assert dict(projection.players["p01"]) == {
        "id": "p01", "role": "seer", "alive": True, "faction": "good",
    }
    with pytest.raises(TypeError):
        projection.players["p01"]["role"] = "werewolf"
    with pytest.raises(AttributeError):
        projection.events[0]["payload"]["items"].append("after")
    mutable = projection.to_mapping()
    mutable["events"][0]["payload"]["items"].append("after")
    assert tuple(projection.events[0]["payload"]["items"]) == ("before",)
    assert "SECRET" not in json.dumps(mutable, ensure_ascii=False)
    json.dumps(mutable, ensure_ascii=False)


def test_direct_projection_constructor_redacts_player_extras() -> None:
    from werewolf_agent.evaluation.game_projection import AcceptanceGameProjection

    projection = AcceptanceGameProjection(
        game_id="g-direct-private-player",
        players={"p01": {
            "id": "p01", "role": "seer", "alive": True, "faction": "good",
            "private_prompt": "SECRET",
        }},
        events=(),
        winning_faction=None,
        status="running",
    )

    assert dict(projection.players["p01"]) == {
        "id": "p01", "role": "seer", "alive": True, "faction": "good",
    }
    assert "SECRET" not in json.dumps(projection.to_mapping(), ensure_ascii=False)


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ({"game_id": "g", "players": {"p": {"role": "villager"}}}, "missing_events"),
        ({"game_id": "g", "players": {"p": {"role": "villager"}}, "events": {}},
         "invalid_events_container"),
        ({"game_id": "g", "players": {"p": {"role": "villager"}}, "events": ["bad"]},
         "invalid_event_entry"),
        ({"game_id": "g", "players": {"p": {"role": "villager"}},
          "events": [{"type": "audit", "payload": {"custom": object()}}]},
         "invalid_event_payload"),
    ],
)
def test_invalid_event_inputs_fail_closed_with_stable_reason(source, reason) -> None:
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    projection = project_acceptance_game(source)

    assert projection.supported is False
    assert projection.unsupported_reason == reason


def test_state_with_custom_event_payload_fails_closed() -> None:
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    state = GameState(
        game_id="g-state-invalid-event",
        players={"p01": PlayerState(id="p01", role="villager")},
        events=[GameEvent(type="audit", payload={"custom": object()})],
    )

    projection = project_acceptance_game(state)

    assert projection.supported is False
    assert projection.unsupported_reason == "invalid_event_payload"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"game_id": object()}, "invalid_game_id"),
        ({"status": "complete"}, "invalid_status"),
        ({"status": []}, "invalid_status"),
        ({"winning_faction": object()}, "invalid_winning_faction"),
        ({"events": "bad"}, "invalid_events_container"),
        ({"events": ({"payload": {}},)}, "invalid_event_entry"),
        ({"events": ({"type": "audit", "payload": []},)}, "invalid_event_entry"),
        ({"players": []}, "invalid_players_container"),
        ({"players": {"p01": "bad"}}, "invalid_player_entry"),
        ({"deaths": {}}, "invalid_deaths_container"),
        ({"metadata": []}, "invalid_metadata_container"),
    ],
)
def test_direct_projection_constructor_fails_closed_for_invalid_structure(
    overrides, reason,
) -> None:
    from werewolf_agent.evaluation.game_projection import AcceptanceGameProjection

    values = {
        "game_id": "g-direct-invalid",
        "events": (),
        "players": {"p01": {"id": "p01", "role": "villager"}},
        "winning_faction": None,
        "status": "running",
    }
    values.update(overrides)

    projection = AcceptanceGameProjection(**values)

    assert projection.supported is False
    assert projection.unsupported_reason == reason
    json.dumps(projection.to_mapping(), ensure_ascii=False)


@pytest.mark.parametrize("source_kind", ["mapping", "state", "projection"])
def test_cyclic_event_payload_fails_closed_without_recursion_error(source_kind) -> None:
    from werewolf_agent.evaluation.game_projection import (
        AcceptanceGameProjection,
        project_acceptance_game,
    )

    cyclic = {}
    cyclic["self"] = cyclic
    event = {"type": "audit", "payload": cyclic}
    players = {"p01": {"id": "p01", "role": "villager"}}
    if source_kind == "mapping":
        source = {"game_id": "g-cycle-map", "players": players, "events": [event]}
    elif source_kind == "state":
        source = GameState(
            game_id="g-cycle-state",
            players={"p01": PlayerState(id="p01", role="villager")},
            events=[GameEvent(type="audit", payload=cyclic)],
        )
    else:
        source = AcceptanceGameProjection(
            game_id="g-cycle-direct",
            players=players,
            events=(event,),
            winning_faction=None,
            status="running",
        )

    projection = project_acceptance_game(source)

    assert projection.supported is False
    assert projection.unsupported_reason == "cyclic_json_value"
    assert "self" not in json.dumps(projection.to_mapping(), ensure_ascii=False)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (lambda: {"nested": _nested_json_value(40)}, "json_depth_exceeded"),
        (lambda: {"items": list(range(10001))}, "json_item_limit_exceeded"),
    ],
)
def test_projection_json_bounds_fail_closed(payload, reason) -> None:
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    projection = project_acceptance_game({
        "game_id": "g-json-bounds",
        "players": {"p01": {"role": "villager"}},
        "events": [{"type": "audit", "payload": payload()}],
    })

    assert projection.supported is False
    assert projection.unsupported_reason == reason


def _nested_json_value(depth: int):
    value = "leaf"
    for _ in range(depth):
        value = [value]
    return value


def test_path_is_only_coerced_for_source_path_metadata() -> None:
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    source_path = project_acceptance_game({
        "game_id": "g-path-metadata",
        "players": {"p01": {"role": "villager"}},
        "events": [],
        "__source_path": Path("legacy/game.json"),
    })
    event_path = project_acceptance_game({
        "game_id": "g-path-event",
        "players": {"p01": {"role": "villager"}},
        "events": [{"type": "audit", "payload": {"path": Path("private")}}],
    })

    assert source_path.metadata["__source_path"] == str(Path("legacy/game.json"))
    assert event_path.supported is False
    assert event_path.unsupported_reason == "invalid_event_payload"


def test_custom_state_metadata_and_players_fail_closed_without_repr_leak() -> None:
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    secret = "PRIVATE_OBJECT_REPR"

    class CustomValue:
        def __repr__(self):
            return secret

    state = SimpleNamespace(
        game_id="g-custom-state",
        status="running",
        winning_faction=None,
        termination_reason=None,
        players={"p01": SimpleNamespace(
            id="p01", role="villager", alive=True, faction=CustomValue(),
        )},
        events=[],
        deaths=[],
        phase="day",
        day_number=1,
        night_number=0,
        hybrid_result=CustomValue(),
    )

    projection = project_acceptance_game(state)
    encoded = json.dumps(projection.to_mapping(), ensure_ascii=False)

    assert projection.supported is False
    assert projection.unsupported_reason in {"invalid_player_value", "invalid_metadata_value"}
    assert secret not in encoded


def test_custom_explicit_unsupported_reason_uses_stable_fallback() -> None:
    from werewolf_agent.evaluation.game_projection import project_acceptance_game

    class CustomReason:
        def __str__(self):
            return "PRIVATE_REASON"

    projection = project_acceptance_game({
        "game_id": "g-custom-reason",
        "players": {"p01": {"role": "villager"}},
        "events": [],
        "_acceptance_projection_supported": False,
        "_acceptance_projection_unsupported_reason": CustomReason(),
    })

    assert projection.supported is False
    assert projection.unsupported_reason == "unsupported_projection"
    assert "PRIVATE_REASON" not in json.dumps(projection.to_mapping(), ensure_ascii=False)


def test_domain_apis_accept_invalid_direct_projection_without_throwing() -> None:
    from werewolf_agent.evaluation.acceptance_power_metrics import (
        compute_power_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_reflection_metrics import (
        compute_reflection_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_terminal_semantic_metrics import (
        compute_terminal_semantic_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_world_metrics import (
        compute_world_acceptance_metrics,
    )
    from werewolf_agent.evaluation.game_projection import AcceptanceGameProjection

    invalid = AcceptanceGameProjection(
        game_id="g-direct-domain",
        players={"p01": {"role": "villager"}},
        events="bad",
        winning_faction=None,
        status="running",
    )
    results = [
        (compute_world_acceptance_metrics([invalid]), "possible_world_metrics_unsupported_reason"),
        (compute_power_acceptance_metrics([invalid]), "power_role_evidence_metrics_unsupported_reason"),
        (compute_reflection_acceptance_metrics([invalid]), "reflection_contamination_metrics_unsupported_reason"),
        (compute_terminal_semantic_acceptance_metrics([invalid]), "semantic_repair_metrics_unsupported_reason"),
    ]

    for metrics, key in results:
        assert metrics[key] == "invalid_events_container"


def test_combined_acceptance_entry_normalizes_each_game_once(monkeypatch) -> None:
    from werewolf_agent.evaluation import game_projection
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    source = {
        "game_id": "g-normalize-once",
        "players": {"p01": {"role": "villager"}},
        "events": [],
    }
    original = game_projection.project_acceptance_game
    calls = []

    def counting_project(value, **kwargs):
        calls.append(value)
        return original(value, **kwargs)

    monkeypatch.setattr(game_projection, "project_acceptance_game", counting_project)

    compute_acceptance_audit_metrics([source])

    assert calls == [source]


def test_normalized_acceptance_games_are_deeply_immutable() -> None:
    from werewolf_agent.evaluation.game_projection import normalize_acceptance_games

    source = {
        "game_id": "g-immutable-normalized",
        "players": {"p01": {"role": "villager"}},
        "events": [{
            "type": "model_execution_audit",
            "payload": {"labels": ["original"]},
        }],
    }

    normalized = normalize_acceptance_games([source])

    with pytest.raises(TypeError):
        normalized[0] = normalized[0]
    with pytest.raises(AttributeError):
        normalized.append(normalized[0])
    with pytest.raises(TypeError):
        normalized[0]["events"] = ()
    with pytest.raises(TypeError):
        normalized[0]["events"][0]["payload"]["labels"][0] = "tampered"

    source["events"][0]["payload"]["labels"][0] = "source-mutated"
    assert normalized[0]["events"][0]["payload"]["labels"] == ("original",)


@pytest.mark.parametrize("container_factory", [list, tuple])
def test_plain_list_or_tuple_projection_markers_are_revalidated(
    container_factory,
) -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )
    from werewolf_agent.evaluation.acceptance_power_metrics import (
        compute_power_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_reflection_metrics import (
        compute_reflection_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_terminal_semantic_metrics import (
        compute_terminal_semantic_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_world_metrics import (
        compute_world_acceptance_metrics,
    )
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    forged = {
        "game_id": "g-forged-normalized-container",
        "players": {"p01": {"role": "villager"}},
        "events": {},
        "_acceptance_projection_supported": True,
        "_acceptance_projection_unsupported_reason": None,
    }

    games = container_factory([forged])
    results = [
        (compute_acceptance_audit_metrics(games), "acceptance_projection_unsupported_reason"),
        (compute_world_acceptance_metrics(games), "possible_world_metrics_unsupported_reason"),
        (compute_power_acceptance_metrics(games), "power_role_evidence_metrics_unsupported_reason"),
        (compute_reflection_acceptance_metrics(games), "reflection_contamination_metrics_unsupported_reason"),
        (compute_terminal_semantic_acceptance_metrics(games), "semantic_repair_metrics_unsupported_reason"),
        (compute_balance_audit(games), "acceptance_projection_unsupported_reason"),
    ]

    for metrics, reason_key in results:
        assert metrics[reason_key] == "invalid_events_container"


def test_projection_module_exposes_no_nominal_trust_token_or_type() -> None:
    from werewolf_agent.evaluation import game_projection

    assert not hasattr(game_projection, "_NORMALIZED_GAMES_CONSTRUCTION_TOKEN")
    assert not hasattr(game_projection, "_NormalizedAcceptanceGames")


@pytest.mark.parametrize("construction", ["module_token", "tuple_new"])
def test_all_public_acceptance_apis_revalidate_nominal_tuple_forgery(
    construction,
) -> None:
    from werewolf_agent.evaluation import game_projection
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )
    from werewolf_agent.evaluation.acceptance_power_metrics import (
        compute_power_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_reflection_metrics import (
        compute_reflection_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_terminal_semantic_metrics import (
        compute_terminal_semantic_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_world_metrics import (
        compute_world_acceptance_metrics,
    )
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    forged = {
        "game_id": "g-forged-private-container",
        "players": {"p01": {"role": "villager"}},
        "events": {},
        "_acceptance_projection_supported": True,
        "_acceptance_projection_unsupported_reason": None,
    }
    nominal_type = getattr(game_projection, "_NormalizedAcceptanceGames", None)
    if nominal_type is None:
        nominal_type = type("CallerForgedNormalizedGames", (tuple,), {})
    if (
        construction == "module_token"
        and hasattr(game_projection, "_NORMALIZED_GAMES_CONSTRUCTION_TOKEN")
    ):
        games = nominal_type(
            (forged,),
            _token=game_projection._NORMALIZED_GAMES_CONSTRUCTION_TOKEN,
        )
    else:
        games = tuple.__new__(nominal_type, (forged,))

    results = [
        (compute_acceptance_audit_metrics(games), "acceptance_projection_unsupported_reason"),
        (compute_world_acceptance_metrics(games), "possible_world_metrics_unsupported_reason"),
        (compute_power_acceptance_metrics(games), "power_role_evidence_metrics_unsupported_reason"),
        (compute_reflection_acceptance_metrics(games), "reflection_contamination_metrics_unsupported_reason"),
        (compute_terminal_semantic_acceptance_metrics(games), "semantic_repair_metrics_unsupported_reason"),
        (compute_balance_audit(games), "acceptance_projection_unsupported_reason"),
    ]

    for metrics, reason_key in results:
        assert metrics[reason_key] == "invalid_events_container"


@pytest.mark.parametrize(
    "domain_import",
    [
        ("acceptance_world_metrics", "compute_world_acceptance_metrics"),
        ("acceptance_power_metrics", "compute_power_acceptance_metrics"),
        ("acceptance_reflection_metrics", "compute_reflection_acceptance_metrics"),
        (
            "acceptance_terminal_semantic_metrics",
            "compute_terminal_semantic_acceptance_metrics",
        ),
    ],
)
def test_direct_domain_entry_normalizes_each_game_once(
    monkeypatch, domain_import,
) -> None:
    from importlib import import_module

    from werewolf_agent.evaluation import game_projection

    source = {
        "game_id": "g-direct-domain-normalize-once",
        "players": {"p01": {"role": "villager"}},
        "events": [],
    }
    original = game_projection.project_acceptance_game
    calls = []

    def counting_project(value, **kwargs):
        calls.append(value)
        return original(value, **kwargs)

    monkeypatch.setattr(game_projection, "project_acceptance_game", counting_project)
    module_name, function_name = domain_import
    function = getattr(
        import_module(f"werewolf_agent.evaluation.{module_name}"), function_name,
    )

    function([source])

    assert calls == [source]


def test_combined_and_domain_metrics_accept_immutable_normalized_games() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )
    from werewolf_agent.evaluation.acceptance_power_metrics import (
        compute_power_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_reflection_metrics import (
        compute_reflection_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_terminal_semantic_metrics import (
        compute_terminal_semantic_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_world_metrics import (
        compute_world_acceptance_metrics,
    )
    from werewolf_agent.evaluation.game_projection import normalize_acceptance_games

    normalized = normalize_acceptance_games([{
        "game_id": "g-immutable-domains",
        "players": {"p01": {"role": "villager"}},
        "events": {},
    }])
    results = [
        (compute_acceptance_audit_metrics(normalized), "acceptance_projection_unsupported_reason"),
        (compute_world_acceptance_metrics(normalized), "possible_world_metrics_unsupported_reason"),
        (compute_power_acceptance_metrics(normalized), "power_role_evidence_metrics_unsupported_reason"),
        (compute_reflection_acceptance_metrics(normalized), "reflection_contamination_metrics_unsupported_reason"),
        (compute_terminal_semantic_acceptance_metrics(normalized), "semantic_repair_metrics_unsupported_reason"),
    ]

    for metrics, reason_key in results:
        assert metrics[reason_key] == "invalid_events_container"


def test_direct_domain_apis_expose_projection_unsupported_reason() -> None:
    from werewolf_agent.evaluation.acceptance_power_metrics import (
        compute_power_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_reflection_metrics import (
        compute_reflection_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_terminal_semantic_metrics import (
        compute_terminal_semantic_acceptance_metrics,
    )
    from werewolf_agent.evaluation.acceptance_world_metrics import (
        compute_world_acceptance_metrics,
    )

    invalid = [{"game_id": "g-invalid", "players": {"p": {"role": "villager"}},
                "events": {}}]
    results = [
        (compute_world_acceptance_metrics(invalid), "possible_world_metrics_unsupported_reason"),
        (compute_power_acceptance_metrics(invalid), "power_role_evidence_metrics_unsupported_reason"),
        (compute_reflection_acceptance_metrics(invalid), "reflection_contamination_metrics_unsupported_reason"),
        (compute_terminal_semantic_acceptance_metrics(invalid), "semantic_repair_metrics_unsupported_reason"),
    ]

    for metrics, key in results:
        assert metrics[key] == "invalid_events_container"


def test_normalized_projection_preserves_unsupported_reason_across_domain_apis() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )
    from werewolf_agent.evaluation.game_projection import (
        normalize_acceptance_games,
        project_acceptance_game,
    )

    invalid = [{
        "game_id": "g-invalid-roundtrip",
        "players": {"p": {"role": "villager"}},
        "events": {},
    }]
    normalized = normalize_acceptance_games(invalid)

    assert project_acceptance_game(normalized[0]).unsupported_reason == (
        "invalid_events_container"
    )
    combined = compute_acceptance_audit_metrics(invalid)
    assert combined["possible_world_metrics_unsupported_reason"] == (
        "invalid_events_container"
    )
    assert combined["power_role_evidence_metrics_unsupported_reason"] == (
        "invalid_events_container"
    )
    assert combined["reflection_contamination_metrics_unsupported_reason"] == (
        "invalid_events_container"
    )
    assert combined["semantic_repair_metrics_unsupported_reason"] == (
        "invalid_events_container"
    )


@pytest.mark.parametrize("game_id", ["../escape", "a/b", "a\\b", "/absolute", "C:\\escape"])
def test_save_game_log_rejects_unsafe_projection_game_id(tmp_path, game_id) -> None:
    from scripts.run_real_game import save_game_log
    from werewolf_agent.evaluation.game_projection import AcceptanceGameProjection

    projection = AcceptanceGameProjection(
        game_id=game_id, events=(), players={}, winning_faction=None, status="running",
    )
    with pytest.raises(ValueError, match="invalid game_id"):
        save_game_log(
            None, 0.1, projection=projection,
            quality_score={"fallback_rate": 0.0, "total_quality_events": 0},
            output_dir=tmp_path,
        )


def test_save_game_log_replace_failure_preserves_old_file_and_cleans_temp(
    tmp_path, monkeypatch,
) -> None:
    from scripts import run_real_game
    from werewolf_agent.evaluation.game_projection import AcceptanceGameProjection

    target = tmp_path / "game_g-atomic.json"
    target.write_text("old-valid", encoding="utf-8")
    projection = AcceptanceGameProjection(
        game_id="g-atomic", events=(), players={}, winning_faction=None, status="running",
    )

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(run_real_game.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        run_real_game.save_game_log(
            None, 0.1, projection=projection,
            quality_score={"fallback_rate": 0.0, "total_quality_events": 0},
            output_dir=tmp_path,
        )

    assert target.read_text(encoding="utf-8") == "old-valid"
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_game_log_atomic_success_leaves_only_final_file(tmp_path) -> None:
    from scripts.run_real_game import save_game_log
    from werewolf_agent.evaluation.game_projection import AcceptanceGameProjection

    projection = AcceptanceGameProjection(
        game_id="g-atomic-ok", events=(), players={}, winning_faction=None, status="running",
    )
    path = save_game_log(
        None, 0.1, projection=projection,
        quality_score={"fallback_rate": 0.0, "total_quality_events": 0},
        output_dir=tmp_path,
    )

    assert json.loads(path.read_text(encoding="utf-8"))["game_id"] == "g-atomic-ok"
    assert list(tmp_path.glob("*.tmp")) == []


def test_low_quality_resolved_directory_cannot_escape_original_output_root(
    tmp_path, monkeypatch,
) -> None:
    from scripts import run_real_game
    from werewolf_agent.evaluation.game_projection import AcceptanceGameProjection

    output_root = tmp_path / "trusted"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_resolve = run_real_game.Path.resolve

    def redirected_resolve(path, *args, **kwargs):
        if path == output_root / "low_quality_games":
            return outside
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(run_real_game.Path, "resolve", redirected_resolve)
    projection = AcceptanceGameProjection(
        game_id="g-low-quality-escape",
        events=(),
        players={},
        winning_faction=None,
        status="running",
    )

    with pytest.raises(ValueError, match="outside output_dir"):
        run_real_game.save_game_log(
            None,
            0.1,
            projection=projection,
            quality_score={"fallback_rate": 1.0, "total_quality_events": 6},
            output_dir=output_root,
        )

    assert list(outside.iterdir()) == []


def test_save_pins_first_resolved_output_root_against_later_root_drift(
    tmp_path, monkeypatch,
) -> None:
    from scripts import run_real_game
    from werewolf_agent.evaluation.game_projection import AcceptanceGameProjection

    requested = tmp_path / "requested"
    canonical = tmp_path / "trusted"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_resolve = run_real_game.Path.resolve
    resolve_calls = []

    def drifting_resolve(path, *args, **kwargs):
        resolve_calls.append(path)
        if path == requested:
            return canonical
        if path == canonical or path.is_relative_to(canonical):
            return outside / path.relative_to(canonical)
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(run_real_game.Path, "resolve", drifting_resolve)
    projection = AcceptanceGameProjection(
        game_id="g-pinned-root",
        events=(),
        players={},
        winning_faction=None,
        status="running",
    )

    with pytest.raises(ValueError, match="outside output_dir"):
        run_real_game.save_game_log(
            None,
            0.1,
            projection=projection,
            quality_score={"fallback_rate": 1.0, "total_quality_events": 6},
            output_dir=requested,
        )

    assert resolve_calls[0] == requested
    assert canonical not in resolve_calls
    assert not list(outside.rglob("game_*.json"))


def test_atomic_writer_rejects_target_parent_outside_trusted_root(tmp_path) -> None:
    from scripts.run_real_game import _atomic_write_json

    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="outside output_dir"):
        _atomic_write_json(
            outside / "game_escape.json",
            {"game_id": "escape"},
            trusted_root=trusted_root,
        )

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("escape_stage", ["target", "temporary"])
def test_atomic_writer_rejects_resolved_target_or_temp_escape(
    tmp_path, monkeypatch, escape_stage,
) -> None:
    from scripts import run_real_game

    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = trusted_root / "game_safe.json"
    real_resolve = run_real_game.Path.resolve

    def redirected_resolve(path, *args, **kwargs):
        if escape_stage == "target" and path == target:
            return outside / path.name
        if escape_stage == "temporary" and path.suffix == ".tmp":
            return outside / path.name
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(run_real_game.Path, "resolve", redirected_resolve)

    with pytest.raises(ValueError, match="outside output_dir"):
        run_real_game._atomic_write_json(
            target,
            {"game_id": "safe"},
            trusted_root=trusted_root,
        )

    assert list(outside.iterdir()) == []
    assert not target.exists()


def test_save_game_log_never_reads_runner(tmp_path) -> None:
    from scripts.run_real_game import save_game_log
    from werewolf_agent.evaluation.game_projection import AcceptanceGameProjection

    class UnreadableRunner:
        def __getattribute__(self, name):
            raise AssertionError(f"runner must not be read: {name}")

    projection = AcceptanceGameProjection(
        game_id="g-no-runner-read",
        events=(),
        players={},
        winning_faction=None,
        status="running",
    )
    path = save_game_log(
        UnreadableRunner(),
        0.1,
        projection=projection,
        quality_score={"fallback_rate": 0.0, "total_quality_events": 0},
        output_dir=tmp_path,
    )

    assert path.name == "game_g-no-runner-read.json"
