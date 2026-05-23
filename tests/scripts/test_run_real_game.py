"""Tests for the real-game runner script reporting helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.run_real_game import print_quality_audit
from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState


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
