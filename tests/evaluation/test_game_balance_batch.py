from __future__ import annotations

import json


def test_balance_audit_flags_high_wolf_win_rate():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    games = [
        {"winning_faction": "werewolf", "events": [], "deaths": []}
        for _ in range(20)
    ]

    audit = compute_balance_audit(games)

    assert audit["wolf_win_rate"] == 1.0
    assert "wolf_win_rate_high" in audit["warnings"]


def test_balance_audit_counts_schema_failures_and_weak_wolf_plan_kills():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    games = [{
        "winning_faction": "werewolf",
        "events": [
            {
                "type": "action_trace_audit",
                "payload": {
                    "action_trace": {
                        "fallback_reason": "fallback: retries exhausted",
                        "parse_error": "Schema validation error",
                    }
                },
            },
            {
                "type": "wolf_team_plan",
                "payload": {
                    "night_number": 1,
                    "night_kill_primary": "p05",
                    "evidence_quality": "none",
                },
            },
            {
                "type": "wolf_kill_selected",
                "payload": {
                    "night_number": 1,
                    "target_id": "p05",
                    "reason": "wolf_team_plan",
                },
            },
        ],
        "deaths": [],
    }]

    audit = compute_balance_audit(games)

    assert audit["fallback_action_rate"] == 1.0
    assert audit["schema_failure_rate"] == 1.0
    assert audit["weak_wolf_plan_kill_count"] == 1
    assert "weak_wolf_plan_kills_present" in audit["warnings"]


def test_load_game_logs_reads_json_files(tmp_path):
    from werewolf_agent.evaluation.balance_audit import load_game_logs

    path = tmp_path / "game.json"
    path.write_text(json.dumps({"winning_faction": "good"}), encoding="utf-8")

    assert load_game_logs([path]) == [{"winning_faction": "good"}]


def test_balance_audit_reads_runtime_vote_list_shape():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    games = [{
        "winning_faction": "good",
        "events": [{
            "type": "vote_resolved",
            "payload": {
                "votes": [
                    {"voter": "p01", "target": "p05"},
                    {"voter": "p02", "target": "p05"},
                    {"voter": "p03", "target": "p06"},
                ],
            },
        }],
    }]

    audit = compute_balance_audit(games)

    assert audit["mean_vote_concentration"] == 2 / 3


def test_balance_audit_counts_any_parse_failure():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    games = [{
        "winning_faction": "good",
        "events": [{
            "type": "action_trace_audit",
            "payload": {
                "action_trace": {
                    "parse_error": "Could not parse JSON object",
                },
            },
        }],
    }]

    audit = compute_balance_audit(games)

    assert audit["schema_failure_rate"] == 1.0
