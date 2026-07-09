from __future__ import annotations

import json


def test_public_fact_claim_helpers_are_split_from_balance_audit_facade():
    from werewolf_agent.evaluation import balance_audit, balance_public_claims

    assert (
        balance_audit._unsupported_public_fact_claim_count
        is balance_public_claims.unsupported_public_fact_claim_count
    )
    assert (
        balance_audit._unsupported_claims_in_text
        is balance_public_claims.unsupported_claims_in_text
    )


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
                "type": "wolf_team_plan_fallback",
                "payload": {"night_number": 1},
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
    assert audit["wolf_team_plan_fallback_rate"] == 1.0
    assert audit["wolf_team_plan_fallback_count"] == 1
    assert audit["wolf_team_plan_count"] == 1
    assert audit["schema_failure_rate"] == 1.0
    assert audit["weak_wolf_plan_kill_count"] == 1
    assert "weak_wolf_plan_kills_present" in audit["warnings"]
    assert "wolf_team_plan_fallback_high" in audit["warnings"]


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


def test_balance_audit_counts_sheriff_and_power_role_guardrails():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    games = [
        {
            "winning_faction": "werewolf",
            "players": {
                "p01": {"role": "werewolf"},
                "p02": {"role": "seer"},
                "p03": {"role": "villager"},
                "p04": {"role": "hunter"},
                "p05": {"role": "idiot"},
            },
            "events": [
                {
                    "type": "sheriff_elected",
                    "payload": {"sheriff_id": "p01"},
                },
                {
                    "type": "action_trace_audit",
                    "payload": {
                        "task_type": "sheriff_vote",
                        "action_traces": {
                            "p02": {"fallback_reason": "fallback: invalid target"},
                            "p03": {},
                        },
                    },
                },
                {
                    "type": "action_trace_audit",
                    "payload": {
                        "task_type": "speech",
                        "action_trace": {
                            "agent_id": "p04",
                            "parse_error": "Could not parse JSON object",
                        },
                    },
                },
                {
                    "type": "wolf_team_plan",
                    "payload": {"night_number": 1, "evidence_quality": "weak"},
                },
                {
                    "type": "wolf_kill_selected",
                    "payload": {
                        "night_number": 1,
                        "target_id": "p03",
                        "reason": "wolf_team_plan",
                    },
                },
            ],
            "deaths": [{"player_id": "p03", "reason": "hunter_shot"}],
        },
        {
            "winning_faction": "good",
            "players": {
                "p01": {"role": "werewolf"},
                "p02": {"role": "seer"},
                "p03": {"role": "villager"},
                "p04": {"role": "hunter"},
                "p05": {"role": "idiot"},
            },
            "events": [
                {
                    "type": "sheriff_elected",
                    "payload": {"sheriff_id": "p03"},
                },
                {
                    "type": "action_trace_audit",
                    "payload": {
                        "task_type": "speech",
                        "action_trace": {"agent_id": "p05"},
                    },
                },
                {
                    "type": "wolf_team_plan",
                    "payload": {"night_number": 1, "evidence_quality": "strong"},
                },
                {
                    "type": "wolf_kill_selected",
                    "payload": {
                        "night_number": 1,
                        "target_id": "p03",
                        "reason": "wolf_team_plan",
                    },
                },
            ],
            "deaths": [{"player_id": "p01", "reason": "hunter_shot"}],
        },
    ]

    audit = compute_balance_audit(games)

    assert audit["sheriff_werewolf_rate"] == 1 / 2
    assert audit["sheriff_vote_fallback_rate"] == 1 / 2
    assert audit["hunter_friendly_fire_rate"] == 1 / 2
    assert audit["weak_plan_kill_rate"] == 1 / 2
    assert audit["power_role_fallback_rate"] == 2 / 3


def test_balance_audit_warns_on_recent_skew_guardrails():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    games = [
        {
            "winning_faction": "werewolf",
            "players": {
                "p01": {"role": "werewolf"},
                "p02": {"role": "seer"},
                "p03": {"role": "villager"},
            },
            "events": [
                {
                    "type": "sheriff_elected",
                    "payload": {"sheriff_id": "p01"},
                },
                {
                    "type": "action_trace_audit",
                    "payload": {
                        "phase": "sheriff_vote",
                        "action_trace": {
                            "agent_id": "p02",
                            "structured_failure_reason": "missing_tool_call",
                        },
                    },
                },
                {
                    "type": "wolf_team_plan",
                    "payload": {"night_number": 1, "evidence_quality": "none"},
                },
                {
                    "type": "wolf_kill_selected",
                    "payload": {
                        "night_number": 1,
                        "target_id": "p03",
                        "reason": "wolf_team_plan",
                    },
                },
            ],
            "deaths": [{"player_id": "p03", "reason": "hunter_shot"}],
        }
        for _ in range(5)
    ]

    audit = compute_balance_audit(games)

    assert "sheriff_werewolf_rate_high" in audit["warnings"]
    assert "sheriff_vote_fallback_high" in audit["warnings"]
    assert "hunter_friendly_fire_high" in audit["warnings"]
    assert "weak_plan_kill_high" in audit["warnings"]


def test_balance_audit_flags_sheriff_werewolf_and_d1_seer_exile():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    game = {
        "winning_faction": "werewolf",
        "players": {
            "p01": {"role": "werewolf"},
            "p02": {"role": "seer"},
        },
        "events": [
            {"type": "sheriff_elected", "payload": {"sheriff_id": "p01"}},
            {
                "type": "vote_resolved",
                "payload": {
                    "day_number": 1,
                    "exiled": "p02",
                    "votes": [
                        {"voter": "p01", "target": "p02", "reason": "x"},
                    ],
                },
            },
        ],
        "deaths": [
            {
                "player_id": "p02",
                "reason": "exile",
                "resolution_batch": "day_1_vote",
            }
        ],
    }

    audit = compute_balance_audit([game])

    assert audit["sheriff_werewolf_rate"] == 1.0
    assert audit["d1_seer_exile_rate"] == 1.0
    assert "sheriff_werewolf_rate_high" in audit["warnings"]


def test_balance_audit_flags_sheriff_vote_and_power_role_fallbacks():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    game = {
        "winning_faction": "good",
        "players": {
            "p01": {"role": "villager"},
            "p02": {"role": "seer"},
        },
        "events": [
            {
                "type": "action_trace_audit",
                "payload": {
                    "player_id": "p01",
                    "phase": "sheriff_vote",
                    "task_type": "sheriff_vote",
                    "action_trace": {
                        "fallback_reason": "fallback: retries exhausted",
                    },
                },
            },
            {
                "type": "action_trace_audit",
                "payload": {
                    "player_id": "p02",
                    "phase": "night",
                    "task_type": "night_action",
                    "action_trace": {
                        "parse_error": "Could not parse JSON object",
                    },
                },
            },
        ],
        "deaths": [],
    }

    audit = compute_balance_audit([game])

    assert audit["sheriff_vote_fallback_rate"] == 1.0
    assert audit["power_role_fallback_rate"] == 1.0
    assert "sheriff_vote_fallback_high" in audit["warnings"]


def test_balance_audit_flags_hunter_friendly_fire():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    game = {
        "winning_faction": "werewolf",
        "players": {
            "p01": {"role": "hunter"},
            "p02": {"role": "witch"},
            "p03": {"role": "werewolf"},
        },
        "events": [],
        "deaths": [
            {
                "player_id": "p02",
                "reason": "hunter_shot",
                "source_player_id": "p01",
            }
        ],
    }

    audit = compute_balance_audit([game])

    assert audit["hunter_friendly_fire_rate"] == 1.0
    assert "hunter_friendly_fire_high" in audit["warnings"]


def test_balance_audit_flags_weak_plan_kill_rate():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    game = {
        "winning_faction": "werewolf",
        "players": {
            "p01": {"role": "werewolf"},
            "p02": {"role": "villager"},
        },
        "events": [
            {
                "type": "wolf_team_plan",
                "payload": {
                    "night_number": 1,
                    "night_kill_primary": "p02",
                    "evidence_quality": "weak",
                },
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
        "deaths": [
            {
                "player_id": "p02",
                "reason": "wolf_kill",
                "resolution_batch": "night_1",
            }
        ],
    }

    audit = compute_balance_audit([game])

    assert audit["weak_plan_kill_rate"] == 1.0
    assert "weak_plan_kill_high" in audit["warnings"]


def test_balance_audit_flags_fallback_plan_kill_without_target_evidence():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    game = {
        "winning_faction": "werewolf",
        "players": {
            "p01": {"role": "werewolf"},
            "p02": {"role": "seer"},
        },
        "events": [
            {
                "type": "wolf_team_plan",
                "payload": {
                    "night_number": 1,
                    "night_kill_primary": "p02",
                    "evidence_quality": "strong",
                    "consensus_method": "fallback",
                    "evidence_from_discussion": [],
                },
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
        "deaths": [
            {
                "player_id": "p02",
                "reason": "wolf_kill",
                "resolution_batch": "night_1",
            }
        ],
    }

    audit = compute_balance_audit([game])

    assert audit["fallback_plan_kill_without_target_evidence_count"] == 1
    assert audit["fallback_plan_kill_without_target_evidence_rate"] == 1.0
    assert "fallback_plan_kill_without_target_evidence_present" in audit["warnings"]


def test_balance_audit_warns_on_recent_small_sample_skew():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    games = [
        {
            "winning_faction": "werewolf",
            "players": {"p01": {"role": "seer"}},
            "events": [
                {
                    "type": "vote_resolved",
                    "payload": {"day_number": 1, "exiled": "p01", "votes": []},
                }
            ],
            "deaths": [{"player_id": "p01", "reason": "exile"}],
        }
        for _ in range(4)
    ]

    audit = compute_balance_audit(games)

    assert audit["wolf_win_rate"] == 1.0
    assert audit["d1_seer_exile_rate"] == 1.0
    assert "wolf_win_rate_high" in audit["warnings"]
    assert "seer_day1_exile_high" in audit["warnings"]


def test_balance_audit_counts_only_night1_witch_wolf_kill_deaths():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    games = [
        {
            "winning_faction": "werewolf",
            "players": {"p01": {"role": "witch"}},
            "events": [],
            "deaths": [
                {
                    "player_id": "p01",
                    "reason": "wolf_kill",
                    "resolution_batch": "night_2",
                }
            ],
        },
        {
            "winning_faction": "good",
            "players": {"p01": {"role": "witch"}},
            "events": [],
            "deaths": [
                {
                    "player_id": "p01",
                    "reason": "wolf_kill",
                    "resolution_batch": "night_1",
                }
            ],
        },
    ]

    audit = compute_balance_audit(games)

    assert audit["witch_night1_death_rate"] == 1 / 2
    assert audit["witch_wolf_kill_death_rate"] == 1.0


def test_balance_audit_counts_template_vote_reasons_and_public_fact_hallucinations():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    game = {
        "winning_faction": "werewolf",
        "players": {
            "p01": {"role": "werewolf"},
            "p02": {"role": "seer"},
            "p03": {"role": "villager"},
        },
        "events": [
            {
                "type": "speech",
                "payload": {
                    "speaker": "p01",
                    "day_number": 1,
                    "text": "p02报我查杀，但这个逻辑不成立。",
                },
            },
            {
                "type": "vote_resolved",
                "payload": {
                    "day_number": 1,
                    "exiled": "p02",
                    "votes": [
                        {
                            "voter": "p01",
                            "target": "p02",
                            "reason": "p02是当前合法投票候选，需要基于发言、票型和站边继续施压",
                        },
                        {
                            "voter": "p03",
                            "target": "p02",
                            "reason": "p01已自认狼人，p02查杀形成印证。",
                        },
                    ],
                },
            },
        ],
        "deaths": [],
    }

    audit = compute_balance_audit([game])

    assert audit["template_vote_reason_count"] == 1
    assert audit["template_vote_reason_rate"] == 1 / 2
    assert audit["unsupported_public_fact_claim_count"] == 1
    assert "template_vote_reason_high" in audit["warnings"]
    assert "unsupported_public_fact_claims_present" in audit["warnings"]
