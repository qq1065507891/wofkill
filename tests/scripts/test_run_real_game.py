"""Tests for the real-game runner script reporting helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.run_real_game import print_quality_audit
from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState


def test_report_helpers_are_split_from_run_real_game_facade() -> None:
    from scripts import run_real_game, run_real_game_reports

    assert run_real_game.print_game_summary is run_real_game_reports.print_game_summary
    assert run_real_game.print_usage_stats is run_real_game_reports.print_usage_stats
    assert run_real_game.print_pace_report is run_real_game_reports.print_pace_report
    assert run_real_game.print_quality_audit is run_real_game_reports.print_quality_audit
    assert run_real_game.check_leakage is run_real_game_reports.check_leakage


def test_quality_audit_handles_vote_trace_without_parsed_action(capsys) -> None:
    runner = SimpleNamespace(
        state=GameState(
            game_id="g_trace_none",
            phase="day",
            events=[
                GameEvent(
                    type="action_trace_audit",
                    payload={
                        "phase": "vote",
                        "action_trace": {
                            "parsed_action": None,
                            "fallback_reason": "model_failed",
                        },
                    },
                )
            ],
        )
    )

    print_quality_audit(runner)

    out = capsys.readouterr().out
    assert "Votes without basis" in out
    assert "Fallbacks:" in out


def test_format_api_key_status_does_not_expose_key_material() -> None:
    from scripts.run_real_game import _format_api_key_status

    key = "sk-test-1234567890abcdef"

    status = _format_api_key_status(key)

    assert status == "configured"
    assert key[:8] not in status


def test_save_game_log_exports_complete_death_fields(tmp_path, monkeypatch) -> None:
    from scripts import run_real_game

    gs = GameState(
        game_id="g_export",
        players={"hunter": PlayerState(id="hunter", role="hunter", alive=False)},
        deaths=[
            Death(
                player_id="hunter",
                reason="exile",
                timing="day_vote",
                resolution_batch="day_3_vote",
                source_player_id=None,
                can_leave_last_words=True,
                triggered_skills=["hunter_shot"],
            ),
        ],
    )
    runner = SimpleNamespace(game_id="g_export", state=gs, step_count=1)
    monkeypatch.setattr(run_real_game, "ROOT", tmp_path)

    path = run_real_game.save_game_log(runner, elapsed=1.2)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["deaths"] == [
        {
            "player_id": "hunter",
            "reason": "exile",
            "timing": "day_vote",
            "resolution_batch": "day_3_vote",
            "source_player_id": None,
            "can_leave_last_words": True,
            "triggered_skills": ["hunter_shot"],
        },
    ]


def test_quality_score_counts_wolf_team_plan_fallbacks() -> None:
    from scripts import run_real_game

    gs = GameState(
        game_id="g_quality_wolf_plan",
        events=[
            GameEvent(
                type="action_trace_audit",
                payload={"action_trace": {"fallback_reason": "empty_response"}},
            ),
            GameEvent(type="wolf_team_plan_fallback", payload={"night_number": 1}),
            GameEvent(type="wolf_team_plan", payload={"night_number": 1}),
        ],
    )
    runner = SimpleNamespace(state=gs, step_count=3)

    quality = run_real_game.compute_game_quality_score(runner)

    assert quality["action_fallback_count"] == 1
    assert quality["wolf_team_plan_fallback_count"] == 1
    assert quality["fallback_count"] == 2
    assert quality["total_wolf_team_plans"] == 1
    assert quality["total_quality_events"] == 2
    assert quality["fallback_rate"] == 1.0


def test_save_game_log_exports_hybrid_fields_from_victory_event(tmp_path, monkeypatch) -> None:
    from scripts import run_real_game

    gs = GameState(
        game_id="g_hybrid_export",
        players={
            "p01": PlayerState(id="p01", role="werewolf", alive=True),
            "p12": PlayerState(id="p12", role="hybrid", alive=True),
        },
        winning_faction="werewolf",
        events=[
            GameEvent(
                type="victory",
                payload={
                    "winner": "werewolf",
                    "hybrid_master_id": "p01",
                    "hybrid_master_faction": "werewolf",
                    "hybrid_result": "win",
                },
            ),
        ],
    )
    runner = SimpleNamespace(game_id="g_hybrid_export", state=gs, step_count=1)
    monkeypatch.setattr(run_real_game, "ROOT", tmp_path)

    path = run_real_game.save_game_log(runner, elapsed=1.2)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["hybrid_master_id"] == "p01"
    assert data["hybrid_master_faction"] == "werewolf"
    assert data["hybrid_result"] == "win"
