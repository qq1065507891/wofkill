from __future__ import annotations

import json


def test_recent_balance_report_includes_new_guardrails(tmp_path):
    from scripts.analyze_recent_balance import build_recent_balance_report

    game = {
        "winning_faction": "werewolf",
        "players": {
            "p01": {"role": "werewolf"},
            "p02": {"role": "seer"},
        },
        "events": [
            {"type": "sheriff_elected", "payload": {"sheriff_id": "p01"}},
            {
                "type": "action_trace",
                "payload": {
                    "phase": "sheriff_vote",
                    "action_trace": {"fallback_reason": "parse_error"},
                },
            },
            {
                "type": "wolf_team_plan_fallback",
                "payload": {"night_number": 1},
            },
            {
                "type": "wolf_team_plan",
                "payload": {"night_number": 1, "evidence_quality": "weak"},
            },
            {
                "type": "wolf_kill_selected",
                "payload": {
                    "night_number": 1,
                    "target_id": "p02",
                    "reason": "wolf_team_plan",
                },
            },
        ],
        "deaths": [{"player_id": "p02", "reason": "hunter_shot"}],
    }
    path = tmp_path / "game_g_recent.json"
    path.write_text(json.dumps(game), encoding="utf-8")

    report = build_recent_balance_report([path])

    assert report["persona_prompt_confirmation"] == {
        "supported": False,
        "configured_action_count": 0,
        "confirmed_action_count": 0,
        "confirmation_rate": None,
    }

    assert report["sheriff_werewolf_rate"] == 1.0
    assert report["sheriff_vote_fallback_rate"] == 1.0
    assert report["wolf_team_plan_fallback_rate"] == 1.0
    assert report["weak_plan_kill_rate"] == 1.0
    assert "hunter_friendly_fire_rate" in report
