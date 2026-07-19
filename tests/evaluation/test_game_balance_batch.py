# -*- coding: utf-8 -*-
"""验证多局平衡审计与公开声明安全修复。

作者: Project contributors
修改日期: 2026-07-16
"""

from __future__ import annotations

import json

import pytest


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


def test_public_fact_guard_redacts_unsupported_night_claims():
    from werewolf_agent.evaluation.balance_public_claims import sanitize_public_text

    text = "p04知道首夜刀口信息，所以我不信任p04。"
    sanitized, count = sanitize_public_text(text, [])

    assert count == 1
    assert "p04知道首夜刀口信息" not in sanitized
    assert "未公开事实" not in sanitized
    assert sanitized == "对p04的身份声明暂不采信，需继续核验，所以我不信任p04。"


def test_public_fact_guard_preserves_supported_role_claims():
    from werewolf_agent.evaluation.balance_public_claims import sanitize_public_text

    history = [("p02", "我是预言家，首夜查验p04为好人。")]
    text = "p02自称预言家，应该优先核对他的警徽流。"

    sanitized, count = sanitize_public_text(text, history)

    assert count == 0
    assert sanitized == text


def test_public_fact_guard_repairs_only_invalid_mixed_claim() -> None:
    from werewolf_agent.evaluation.balance_public_claims import sanitize_public_text

    history = [("p05", "我是女巫，但暂时不公开药水。")]
    text = "我怀疑p08；p05刚才声称自己是女巫；系统确认p08是狼人。"
    sanitized, count = sanitize_public_text(text, history)
    assert count == 1
    assert "p05刚才声称自己是女巫" in sanitized
    assert "p08" in sanitized
    assert "系统确认p08是狼人" not in sanitized
    assert "我怀疑p08" in sanitized


def test_public_fact_guard_preserves_negated_system_fact() -> None:
    from werewolf_agent.evaluation.balance_public_claims import sanitize_public_text

    text = "我并不认为系统确认p08是狼人，我只是怀疑p08。"
    sanitized, count = sanitize_public_text(text, [])
    assert (sanitized, count) == (text, 0)


def test_public_claim_classifier_separates_semantic_types() -> None:
    from werewolf_agent.evaluation.balance_public_claims import classify_public_claims

    claims = classify_public_claims(
        "p05声称自己是女巫；系统确认p08是狼人；我认为p09更可疑。"
    )

    assert [claim.claim_type.value for claim in claims] == [
        "player_claim",
        "system_fact",
        "current_player_inference",
    ]


@pytest.mark.parametrize("relation", ["是", "不是", "并非", "不为"])
def test_public_claim_classifier_preserves_direct_attributed_role_polarity(
    relation: str,
) -> None:
    from werewolf_agent.evaluation.balance_public_claims import classify_public_claims

    claims = classify_public_claims(f"p03声称p05{relation}狼人")

    assert len(claims) == 1
    claim = claims[0]
    assert claim.support_kind == "role_assignment"
    assert claim.speaker_attribution == "p03"
    assert claim.target == "p05"
    assert claim.role == "狼人"
    assert claim.negated is (relation != "是")


@pytest.mark.parametrize(
    ("claim", "history", "expected_unsupported"),
    [
        ("p03声称p05是狼人", [("p03", "我不认为p05是狼人")], 1),
        ("p03声称p05不是狼人", [("p03", "我不认为p05不是狼人")], 0),
        ("p05声称自己是预言家", [("p05", "我没有说我是预言家")], 1),
        ("p05声称自己不是预言家", [("p05", "我没有说我不是预言家")], 0),
        ("p03声称p05是狼人", [("p03", "我认为p05是狼人")], 0),
        ("p05声称自己是预言家", [("p05", "我声称自己是预言家")], 0),
    ],
)
def test_public_sanitizer_matches_v2_discourse_polarity(
    claim: str,
    history: list[tuple[str, str]],
    expected_unsupported: int,
) -> None:
    from werewolf_agent.evaluation.balance_public_claims import sanitize_public_text
    from werewolf_agent.agents.schemas import (
        ActionType,
        AgentContext,
        SpeechPlayerAction,
        TaskType,
    )
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    sanitized, unsupported = sanitize_public_text(claim, history)
    context = AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p02"],
        public_claim_ledger=[{"speaker": speaker, "text": text} for speaker, text in history],
    )
    source = SpeechPlayerAction(
        target_id="p02", speech="我怀疑p02。", reason="test", confidence=0.5
    )
    final = SpeechPlayerAction(
        target_id="p02", speech=claim, reason="test", confidence=0.5
    )
    v2 = validate_semantic_repair(context, source, final)

    assert unsupported == expected_unsupported
    assert (unsupported > 0) == ("unsupported_public_claim" in v2.reason_codes)
    if expected_unsupported:
        assert sanitized != claim
    else:
        assert sanitized == claim


def test_public_fact_guard_is_driven_by_authoritative_classifier(monkeypatch) -> None:
    from werewolf_agent.evaluation import balance_public_claims as claims_module

    text = "裁判锚定p05；我仍然怀疑p05。"
    classified = claims_module.ClassifiedPublicClaim(
        claims_module.PublicClaimType.SYSTEM_FACT,
        "裁判锚定p05",
        0,
        len("裁判锚定p05"),
        target="p05",
    )
    monkeypatch.setattr(claims_module, "classify_public_claims", lambda _: [classified])

    sanitized, count = claims_module.sanitize_public_text(text, [])

    assert count == 1
    assert sanitized == "对p05的身份声明暂不采信，需继续核验；我仍然怀疑p05。"


@pytest.mark.parametrize(
    "text",
    [
        "p05声称自己是女巫，所以我投p05。",
        "p05声称知道首夜刀口信息，所以我投p05。",
    ],
)
def test_public_fact_guard_keeps_target_in_zero_support_repair(text: str) -> None:
    from werewolf_agent.evaluation.balance_public_claims import sanitize_public_text

    sanitized, count = sanitize_public_text(text, [])

    assert count == 1
    assert "p05" in sanitized
    assert "投p05" in sanitized
    assert "[未公开事实]" not in sanitized
    assert "对p05的身份声明暂不采信" in sanitized


def test_public_fact_guard_resolves_overlapping_claim_spans_once() -> None:
    from werewolf_agent.evaluation.balance_public_claims import sanitize_public_text

    text = "p04声称自己是女巫并表示知道狼刀信息，所以投p04。"
    sanitized, count = sanitize_public_text(text, [])

    assert count == 1
    assert sanitized.count("对p04的身份声明暂不采信") == 1
    assert "投p04" in sanitized


def test_public_fact_guard_repairs_non_overlapping_claims_stably() -> None:
    from werewolf_agent.evaluation.balance_public_claims import sanitize_public_text

    text = "p04声称自己是女巫；p05声称知道狼刀信息；所以投p04。"
    sanitized, count = sanitize_public_text(text, [])

    assert count == 2
    assert sanitized.count("暂不采信") == 2
    assert "投p04" in sanitized


@pytest.mark.parametrize(
    "text",
    [
        "不能否认系统确认p08是狼人。",
        "没有理由不信系统确认p08是狼人。",
    ],
)
def test_public_fact_guard_treats_double_negation_as_assertive(text: str) -> None:
    from werewolf_agent.evaluation.balance_public_claims import unsupported_claims_in_text

    assert unsupported_claims_in_text(text, []) == 1


@pytest.mark.parametrize(
    "text",
    [
        "系统没有确认p08是狼人。",
        "不能说系统确认p08是狼人。",
    ],
)
def test_public_fact_guard_accepts_authoritative_negation_scope(text: str) -> None:
    from werewolf_agent.evaluation.balance_public_claims import unsupported_claims_in_text

    assert unsupported_claims_in_text(text, []) == 0


def test_balance_audit_flags_high_wolf_win_rate():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    games = [
        {"winning_faction": "werewolf", "events": [], "deaths": []}
        for _ in range(20)
    ]

    audit = compute_balance_audit(games)

    assert audit["wolf_win_rate"] == 1.0
    assert "wolf_win_rate_high" in audit["warnings"]


def test_balance_excludes_aborted_gameplay_but_stability_counts_all_games():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    finished = {
        "game_id": "g-finished",
        "status": "finished",
        "winning_faction": "good",
        "players": {"p01": {"role": "villager"}},
        "events": [{
            "type": "vote_resolved",
            "payload": {"votes": [
                {"target": "p02"}, {"target": "p02"},
            ]},
        }],
        "deaths": [],
    }
    aborted = {
        "game_id": "g-aborted",
        "status": "aborted",
        "termination_reason": "step_limit",
        "players": {"p01": {"role": "seer"}},
        "events": [
            {
                "type": "vote_resolved",
                "payload": {
                    "day_number": 1,
                    "exiled": "p01",
                    "votes": [{"target": "p01"}, {"target": "p02"}],
                },
            },
            {
                "type": "action_trace_audit",
                "payload": {"action_trace": {
                    "fallback_reason": "runtime abort fallback",
                    "parse_error": "runtime abort schema error",
                }},
            },
        ],
        "deaths": [],
    }

    audit = compute_balance_audit([finished, aborted])

    assert audit["games"] == 2
    assert audit["completed_game_count"] == 1
    assert audit["aborted_game_count"] == 1
    assert audit["completion_rate"] == 0.5
    assert audit["good_win_rate"] == 1.0
    assert audit["wolf_win_rate"] == 0.0
    assert audit["mean_vote_concentration"] == 1.0
    assert audit["seer_day1_exile_rate"] == 0.0
    assert audit["fallback_action_rate"] == 0.0
    assert audit["schema_failure_rate"] == 0.0


def test_aborted_acceptance_evidence_does_not_enter_gameplay_metrics():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    aborted = {
        "game_id": "g-aborted-only",
        "status": "aborted",
        "termination_reason": "unrecoverable_runtime_error",
        "players": {
            "p01": {"role": "seer", "alive": True},
            "p02": {"role": "werewolf", "alive": True},
        },
        "events": [
            {
                "type": "action_trace_audit",
                "payload": {
                    "task_type": "seer_check",
                    "action_trace": {
                        "task_type": "seer_check",
                        "final_action_type": "seer_check",
                        "execution_attempts": [{"attempt_outcome": "success"}],
                    },
                },
            },
            {
                "type": "seer_check",
                "payload": {
                    "actor": "p01", "target": "p02", "result": "werewolf",
                },
            },
            {
                "type": "reflection_complete",
                "payload": {"entries": [{
                    "player_id": "p01",
                    "verification": {
                        "status": "verified",
                        "verified_fact_count": 1,
                        "verified_lessons": [{"lesson_id": "l1"}],
                        "rejected_fact_count": 0,
                        "rejected_lesson_count": 0,
                    },
                }]},
            },
            {
                "type": "semantic_repair_audit",
                "payload": {"success": True, "verified_claim_retention_rate": 1.0},
            },
        ],
        "deaths": [],
    }

    audit = compute_balance_audit([aborted])

    assert audit["games"] == 1
    assert audit["completed_game_count"] == 0
    assert audit["aborted_game_count"] == 1
    assert audit["decision_count"] == 0
    assert audit["critical_task_reasoning_status_request_count"] == 0
    assert audit["power_role_evidence_metrics_supported"] is False
    assert audit["reflection_contamination_metrics_supported"] is False
    assert audit["semantic_repair_metrics_supported"] is False
    assert audit["possible_world_metrics_supported"] is False


def test_balance_audit_normalizes_each_game_once(monkeypatch) -> None:
    from werewolf_agent.evaluation import game_projection
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    source = {
        "game_id": "g-balance-normalize-once",
        "players": {"p01": {"role": "villager"}},
        "events": [],
        "deaths": [],
    }
    original = game_projection.project_acceptance_game
    calls = []

    def counting_project(value, **kwargs):
        calls.append(value)
        return original(value, **kwargs)

    monkeypatch.setattr(game_projection, "project_acceptance_game", counting_project)

    compute_balance_audit([source])

    assert calls == [source]


def test_balance_audit_does_not_trust_forged_projection_markers() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    audit = compute_balance_audit([{
        "game_id": "g-forged-projection",
        "players": {"p01": {"role": "villager"}},
        "events": {},
        "deaths": [],
        "_acceptance_projection_supported": True,
        "_acceptance_projection_unsupported_reason": None,
    }])

    assert audit["acceptance_projection_supported"] is False
    assert audit["acceptance_projection_unsupported_reason"] == (
        "no_games"
    )


def test_aborted_invalid_projection_cannot_override_finished_acceptance() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    finished = {
        "game_id": "g-valid-finished",
        "status": "finished",
        "winning_faction": "good",
        "players": {"p01": {"role": "villager", "alive": True}},
        "events": [],
        "deaths": [],
    }
    aborted_invalid = {
        "game_id": "g-invalid-aborted",
        "status": "aborted",
        "termination_reason": "step_limit",
        "players": {"p02": {"role": "seer", "alive": True}},
        "events": {"forged": "container"},
        "deaths": [],
    }

    audit = compute_balance_audit([finished, aborted_invalid])

    assert audit["games"] == 2
    assert audit["completed_game_count"] == 1
    assert audit["aborted_game_count"] == 1
    assert audit["acceptance_projection_supported"] is True
    assert audit["acceptance_projection_unsupported_reason"] is None


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


def test_balance_audit_reports_disjoint_wolf_plan_outcomes_with_exact_denominator():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    games = [{
        "winning_faction": "good",
        "events": [
            {"type": "wolf_team_plan", "payload": {"normalization_repairs": ["truncate:reasoning"]}},
            {"type": "wolf_team_plan_fallback", "payload": {"reason": "schema_validation_failed"}},
            {"type": "wolf_team_plan", "payload": {"consensus_method": "fallback"}},
            {"type": "wolf_team_plan_fallback", "payload": {"reason": "captain_agent_missing"}},
            {"type": "wolf_team_plan", "payload": {"consensus_method": "fallback"}},
        ],
        "deaths": [],
    }]

    audit = compute_balance_audit(games)

    assert audit["wolf_team_plan_outcome_metrics_supported"] is True
    assert audit["wolf_team_plan_total_count"] == 3
    assert audit["wolf_team_plan_normalization_success_count"] == 1
    assert audit["wolf_team_plan_schema_terminal_fallback_count"] == 1
    assert audit["wolf_team_plan_strategy_terminal_fallback_count"] == 0
    assert audit["wolf_team_plan_other_terminal_fallback_count"] == 1
    assert audit["wolf_team_plan_normalization_triggered_count"] == 1
    assert audit["wolf_team_plan_normalization_success_rate"] == 1.0
    assert audit["wolf_team_plan_schema_terminal_fallback_rate"] == 1 / 3
    assert audit["wolf_team_plan_strategy_terminal_fallback_rate"] == 0
    assert audit["wolf_team_plan_other_terminal_fallback_rate"] == 1 / 3


def test_balance_audit_marks_wolf_plan_outcomes_unsupported_without_denominator():
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    audit = compute_balance_audit([{"winning_faction": "good", "events": [], "deaths": []}])

    assert audit["wolf_team_plan_outcome_metrics_supported"] is False
    assert audit["wolf_team_plan_total_count"] == 0
    assert audit["wolf_team_plan_normalization_success_rate"] is None
    assert audit["wolf_team_plan_schema_terminal_fallback_rate"] is None
    assert audit["wolf_team_plan_strategy_terminal_fallback_rate"] is None


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (
            [
                {"type": "wolf_team_plan_fallback", "payload": {"night_number": 1, "reason": "schema_validation_failed"}},
                {"type": "wolf_team_plan_fallback", "payload": {"night_number": 1, "reason": "schema_validation_failed"}},
                {"type": "wolf_team_plan", "payload": {"night_number": 1, "consensus_method": "fallback"}},
            ],
            {"total": 1, "schema": 1, "strategy": 0, "other": 0},
        ),
        (
            [{"type": "wolf_team_plan_fallback", "payload": {"night_number": 2, "reason": "quorum_not_met"}}],
            {"total": 1, "schema": 0, "strategy": 1, "other": 0},
        ),
        (
            [{"type": "wolf_team_plan_fallback", "payload": {"night_number": 3, "reason": "provider_error"}}],
            {"total": 1, "schema": 0, "strategy": 0, "other": 1},
        ),
        (
            [
                {"type": "wolf_team_plan", "payload": {"night_number": 4, "decision_id": "d1"}},
                {"type": "wolf_team_plan", "payload": {"night_number": 4, "decision_id": "d2"}},
            ],
            {"total": 2, "schema": 0, "strategy": 0, "other": 0},
        ),
    ],
)
def test_wolf_plan_outcomes_pair_and_classify_each_decision_once(events, expected):
    from werewolf_agent.evaluation.balance_audit import compute_wolf_plan_outcome_metrics

    metrics = compute_wolf_plan_outcome_metrics([{"game_id": "g1", "events": events}])

    assert metrics["wolf_team_plan_outcome_metrics_supported"] is True
    assert metrics["wolf_team_plan_total_count"] == expected["total"]
    assert metrics["wolf_team_plan_schema_terminal_fallback_count"] == expected["schema"]
    assert metrics["wolf_team_plan_strategy_terminal_fallback_count"] == expected["strategy"]
    assert metrics["wolf_team_plan_other_terminal_fallback_count"] == expected["other"]
    for name, value in metrics.items():
        if name.endswith("_rate") and value is not None:
            assert 0 <= value <= 1


def test_load_game_logs_reads_json_files(tmp_path):
    from werewolf_agent.evaluation.balance_audit import load_game_logs

    path = tmp_path / "game.json"
    path.write_text(json.dumps({"winning_faction": "good"}), encoding="utf-8")

    assert load_game_logs([path]) == [{
        "winning_faction": "good",
        "__source_path": str(path.resolve()),
    }]


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


def test_balance_audit_reads_v2_night_batch_mapping() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    audit = compute_balance_audit(
        [
            {
                "winning_faction": "good",
                "players": {"p01": {"role": "witch"}},
                "events": [],
                "deaths": [
                    {
                        "player_id": "p01",
                        "reason": "wolf_kill",
                        "resolution_batch": {
                            "phase": "night",
                            "number": 1,
                            "cause": "wolf_kill",
                        },
                    }
                ],
            }
        ]
    )

    assert audit["witch_night1_death_rate"] == 1.0


@pytest.mark.parametrize(
    ("death", "expected"),
    [
        (
            {
                "resolution_batch": "night_1",
                "resolution_batch_parse_failed": True,
            },
            None,
        ),
        (
            {
                "resolution_batch": "night_1",
                "resolution_batch_parse_failed": True,
                "night_number": "2",
            },
            2,
        ),
        (
            {
                "resolution_batch": "night_2",
                "resolution_batch_parse_failed": False,
                "night_number": 1,
            },
            2,
        ),
    ],
)
def test_death_night_number_ignores_marked_batch_and_uses_trusted_fallback(
    death: dict[str, object],
    expected: int | None,
) -> None:
    from werewolf_agent.evaluation.balance_audit import _death_night_number

    assert _death_night_number(death) == expected


@pytest.mark.parametrize(
    ("night_number", "expected"),
    [
        (True, None),
        (False, None),
        (-1, None),
        (0, None),
        ("9" * 5_000, None),
        ("١", None),
        (1, 1),
        ("1", 1),
    ],
)
def test_death_night_number_validates_trusted_fallback(
    night_number: object,
    expected: int | None,
) -> None:
    from werewolf_agent.evaluation.balance_audit import _death_night_number

    death = {
        "resolution_batch": "night_9",
        "resolution_batch_parse_failed": True,
        "night_number": night_number,
    }

    assert _death_night_number(death) == expected


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
