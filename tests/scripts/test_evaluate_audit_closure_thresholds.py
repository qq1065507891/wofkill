# -*- coding: utf-8 -*-
"""
验证审查闭环浸泡测试的硬门槛投影与失败落盘顺序。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-19
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest


def _v2_event(
    game_id: str,
    sequence_number: int,
    event_type: str,
    payload: dict[str, object],
    *,
    visibility: str = "moderator_only",
    trace_id: str | None = None,
) -> dict[str, object]:
    return {
        "type": event_type,
        "payload": payload,
        "visibility": visibility,
        "event_id": f"{game_id}:e{sequence_number:06d}",
        "sequence_number": sequence_number,
        "occurred_at": "2026-07-18T00:00:00+00:00",
        "game_id": game_id,
        "trace_id": trace_id,
        "schema_version": "2",
    }


def _closure_game(game_id: str, *, single_wolf: bool) -> dict[str, object]:
    wolf_ids = ["w1"] if single_wolf else ["w1", "w2"]
    events: list[dict[str, object]] = []
    for wolf_id in wolf_ids:
        sequence = len(events)
        event_id = f"{game_id}:e{sequence:06d}"
        events.append(_v2_event(
            game_id,
            sequence,
            "wolf_discussion",
            {
                "wolf_id": wolf_id,
                "night_number": 1,
                "target_stance": {
                    "wolf_id": wolf_id,
                    "target_id": "v1",
                    "stance": "support",
                    "priority": "primary",
                    "round_number": sequence + 1,
                    "source_event_id": event_id,
                },
            },
            visibility="werewolf_team_only",
        ))
    status = "single_wolf" if single_wolf else "majority"
    sequence = len(events)
    events.append(_v2_event(
        game_id,
        sequence,
        "wolf_kill_selected",
        {
            "night_number": 1,
            "target_id": "v1",
            "reason": "wolf_stance_consensus",
            "plan_key": "night_kill_primary",
        },
        visibility="werewolf_team_only",
        trace_id=(
            f"{game_id}:werewolf_team:wolf_consensus:D1:N1:"
            f"wolf_kill_selected:stance:{status}:primary:0"
        ),
    ))
    events.append(_v2_event(
        game_id,
        len(events),
        "reflection_complete",
        {
            "status": "complete",
            "valid_entry_count": 1,
            "entries": [{"player_id": "w1"}],
        },
    ))
    events.append(_v2_event(
        game_id,
        len(events),
        "hunter_shot_resolved",
        {
            "actor_id": "w1",
            "target_id": None,
            "public_result": "declined",
        },
        visibility="public",
    ))
    players = {
        wolf_id: {"id": wolf_id, "role": "werewolf", "alive": True}
        for wolf_id in wolf_ids
    }
    players["v1"] = {"id": "v1", "role": "villager", "alive": True}
    return {
        "game_id": game_id,
        "status": "finished",
        "winning_faction": "werewolf",
        "phase": "finished",
        "players": players,
        "events": events,
        "deaths": [],
        "quality_score": {"probe": game_id},
    }


def _passing_report() -> dict[str, object]:
    return {
        "games": 10,
        "unique_game_id_count": 10,
        "unique_game_artifact_path_count": 10,
        "completion_rate": 1.0,
        "terminal_post_win_game_model_call_count": 0,
        "weak_wolf_plan_kill_count": 0,
        "fallback_plan_kill_without_target_evidence_count": 0,
        "wolf_team_plan_outcome_metrics_supported": True,
        "wolf_team_plan_terminal_fallback_rate": 0.09,
        "wolf_plan_schema_fallback_rate": 0.04,
        "persona_injection_confirmation_metrics_supported": True,
        "persona_injection_confirmation_rate": 1.0,
        "persona_exposure_linkage_metrics_supported": True,
        "persona_exposure_linkage_rate": 1.0,
        "critical_task_reasoning_request_coverage_supported": True,
        "critical_task_reasoning_request_coverage": 1.0,
        "critical_task_reasoning_request_dispatched_metrics_supported": True,
        "critical_task_reasoning_request_dispatched_rate": 1.0,
        "critical_task_reasoning_confirmed_rate": 1.0,
        "critical_task_reasoning_requires_confirmation_review_count": 0,
        "critical_task_reasoning_effective_request_metrics_supported": True,
        "critical_task_reasoning_effective_request_rate": 1.0,
        "critical_task_reasoning_unsupported_count": 0,
        "critical_task_reasoning_not_requested_count": 0,
        "critical_task_reasoning_fallback_disabled_count": 0,
        "critical_task_reasoning_status_metrics_supported": True,
        "critical_task_reasoning_status_explicit_rate": 1.0,
        "reasoning_task_type_missing_count": 0,
        "reasoning_task_type_unknown_count": 0,
        "decision_execution_invalid_sequence_count": 0,
        "reasoning_fallback_disabled_count": 0,
        "unsupported_public_fact_claim_count": 0,
        "reflection_contamination_metrics_supported": True,
        "reflection_persisted_rejected_fact_count": 0,
        "semantic_repair_metrics_supported": True,
        "semantic_repair_target_preservation_rate": 1.0,
        "semantic_repair_speaker_attribution_preservation_rate": 1.0,
        "semantic_repair_negation_preservation_rate": 1.0,
        "semantic_repair_no_new_claim_rate": 1.0,
        "semantic_repair_success_rate": 0.95,
        "semantic_repair_generic_template_count": 0,
        "semantic_repair_verified_claim_retention_metrics_supported": True,
        "semantic_repair_verified_claim_retention_rate": 1.0,
        "semantic_repair_public_evidence_safety_metrics_supported": True,
        "semantic_repair_public_evidence_safety_rate": 1.0,
        "possible_world_metrics_supported": True,
        "possible_world_unique_rate": 1.0,
        "possible_world_evidence_coverage_rate": 1.0,
        "power_role_evidence_metrics_supported": True,
        "power_role_evidence_completeness_rate": 1.0,
        "decision_execution_metrics_supported": True,
        "attempt_retry_consistency_error_count": 0,
        "resolution_batch_integrity_metrics_supported": True,
        "malformed_resolution_batch_count": 0,
        "provider_fallback_route_metrics_supported": True,
        "same_route_provider_fallback_count": 0,
        "saved_offline_quality_consistency_metrics_supported": True,
        "saved_offline_quality_diff_count": 0,
        "terminal_integrity_metrics_supported": True,
        "finished_without_winner_count": 0,
        "abort_terminal_coverage_metrics_supported": True,
        "abort_terminal_coverage_rate": 1.0,
        "wolf_consensus_execution_metrics_supported": True,
        "majority_wolf_kill_execution_metrics_supported": True,
        "majority_wolf_kill_execution_rate": 1.0,
        "single_wolf_kill_execution_metrics_supported": True,
        "single_wolf_kill_execution_rate": 1.0,
        "reflection_transaction_metrics_supported": True,
        "empty_reflection_success_count": 0,
        "source_event_traceability_metrics_supported": True,
        "source_event_id_traceability_rate": 1.0,
        "public_exposure_metrics_supported": True,
        "public_skill_resolution_leak_count": 0,
        "soak_launch_count": 10,
        "soak_finished_count": 10,
        "soak_aborted_count": 0,
        "wolf_win_rate": 0.5,
        "good_win_rate": 0.5,
    }


def test_all_hard_thresholds_pass_with_complete_descriptors() -> None:
    from scripts.evaluate_audit_closure_thresholds import evaluate_thresholds

    result = evaluate_thresholds(_passing_report())

    assert result["overall_pass"] is True
    assert result["requires_more_samples"] is False
    assert result["soak_summary"] == {
        "launch_count": 10,
        "finished_count": 10,
        "aborted_count": 0,
    }
    assert len(result["thresholds"]) == 42
    assert all(set(item) == {
        "name", "supported", "actual", "operator", "expected", "passed"
    } for item in result["thresholds"])


def test_missing_metric_fails_closed_and_boundary_operators_are_exact() -> None:
    from scripts.evaluate_audit_closure_thresholds import evaluate_thresholds

    report = _passing_report()
    report.pop("power_role_evidence_completeness_rate")
    report["wolf_team_plan_terminal_fallback_rate"] = 0.10
    report["wolf_plan_schema_fallback_rate"] = 0.05

    result = evaluate_thresholds(report)
    by_name = {item["name"]: item for item in result["thresholds"]}

    assert by_name["power_role_evidence_completeness_rate"] == {
        "name": "power_role_evidence_completeness_rate",
        "supported": False,
        "actual": None,
        "operator": "==",
        "expected": 1.0,
        "passed": False,
    }
    assert by_name["wolf_team_plan_terminal_fallback_rate"]["passed"] is False
    assert by_name["wolf_plan_schema_fallback_rate"]["passed"] is False
    assert by_name["semantic_repair_success_rate"]["passed"] is True
    assert result["overall_pass"] is False


@pytest.mark.parametrize(
    ("metric_name", "failing_value"),
    [
        ("malformed_resolution_batch_count", 1),
        ("same_route_provider_fallback_count", 1),
        ("saved_offline_quality_diff_count", 1),
        ("finished_without_winner_count", 1),
        ("abort_terminal_coverage_rate", 0.9),
        ("majority_wolf_kill_execution_rate", 0.9),
        ("single_wolf_kill_execution_rate", 0.9),
        ("empty_reflection_success_count", 1),
        ("source_event_id_traceability_rate", 0.9),
        ("public_skill_resolution_leak_count", 1),
    ],
)
def test_post_july14_closure_metric_is_a_hard_gate(
    metric_name: str,
    failing_value: float,
) -> None:
    from scripts.evaluate_audit_closure_thresholds import evaluate_thresholds

    report = _passing_report()
    report[metric_name] = failing_value

    result = evaluate_thresholds(report)
    by_name = {item["name"]: item for item in result["thresholds"]}

    assert by_name[metric_name]["passed"] is False
    assert result["overall_pass"] is False


@pytest.mark.parametrize(
    ("metric_name", "support_name"),
    [
        (
            "malformed_resolution_batch_count",
            "resolution_batch_integrity_metrics_supported",
        ),
        (
            "same_route_provider_fallback_count",
            "provider_fallback_route_metrics_supported",
        ),
        (
            "saved_offline_quality_diff_count",
            "saved_offline_quality_consistency_metrics_supported",
        ),
        ("finished_without_winner_count", "terminal_integrity_metrics_supported"),
        (
            "abort_terminal_coverage_rate",
            "abort_terminal_coverage_metrics_supported",
        ),
        (
            "majority_wolf_kill_execution_rate",
            "majority_wolf_kill_execution_metrics_supported",
        ),
        (
            "single_wolf_kill_execution_rate",
            "single_wolf_kill_execution_metrics_supported",
        ),
        (
            "empty_reflection_success_count",
            "reflection_transaction_metrics_supported",
        ),
        (
            "source_event_id_traceability_rate",
            "source_event_traceability_metrics_supported",
        ),
        (
            "public_skill_resolution_leak_count",
            "public_exposure_metrics_supported",
        ),
    ],
)
def test_post_july14_closure_gate_fails_closed_without_support(
    metric_name: str,
    support_name: str,
) -> None:
    from scripts.evaluate_audit_closure_thresholds import evaluate_thresholds

    report = _passing_report()
    report[support_name] = False

    result = evaluate_thresholds(report)
    by_name = {item["name"]: item for item in result["thresholds"]}

    assert by_name[metric_name]["supported"] is False
    assert by_name[metric_name]["actual"] is None
    assert by_name[metric_name]["passed"] is False
    assert result["overall_pass"] is False


def test_balance_audit_derives_supported_post_july14_closure_metrics() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    games = [
        _closure_game("closure-majority", single_wolf=False),
        _closure_game("closure-single", single_wolf=True),
    ]

    report = compute_balance_audit(
        games,
        quality_recomputer=lambda game: dict(game["quality_score"]),
    )

    assert report["resolution_batch_integrity_metrics_supported"] is True
    assert report["malformed_resolution_batch_count"] == 0
    assert report["provider_fallback_route_metrics_supported"] is True
    assert report["same_route_provider_fallback_count"] == 0
    assert report["saved_offline_quality_consistency_metrics_supported"] is True
    assert report["saved_offline_quality_diff_count"] == 0
    assert report["terminal_integrity_metrics_supported"] is True
    assert report["finished_without_winner_count"] == 0
    assert report["abort_terminal_coverage_metrics_supported"] is True
    assert report["abort_terminal_coverage_rate"] == 1.0
    assert report["wolf_consensus_execution_metrics_supported"] is True
    assert report["majority_wolf_kill_execution_metrics_supported"] is True
    assert report["majority_wolf_kill_execution_rate"] == 1.0
    assert report["single_wolf_kill_execution_metrics_supported"] is True
    assert report["single_wolf_kill_execution_rate"] == 1.0
    assert report["reflection_transaction_metrics_supported"] is True
    assert report["empty_reflection_success_count"] == 0
    assert report["source_event_traceability_metrics_supported"] is True
    assert report["source_event_id_traceability_rate"] == 1.0
    assert report["public_exposure_metrics_supported"] is True
    assert report["public_skill_resolution_leak_count"] == 0


def test_wolf_consensus_execution_rates_are_non_vacuous_per_route() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    game = _closure_game("closure-majority-only", single_wolf=False)
    report = compute_balance_audit(
        [game],
        quality_recomputer=lambda source: dict(source["quality_score"]),
    )

    assert report["majority_wolf_kill_execution_metrics_supported"] is True
    assert report["majority_wolf_kill_execution_rate"] == 1.0
    assert report["single_wolf_kill_execution_metrics_supported"] is False
    assert report["single_wolf_kill_execution_rate"] is None


def test_balance_audit_surfaces_every_decisive_post_july14_failure() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    game = _closure_game("closure-corrupt", single_wolf=False)
    events = game["events"]
    assert isinstance(events, list)
    discussion = events[0]
    stance = discussion["payload"]["target_stance"]
    stance["source_event_id"] = "wrong-source"
    del events[2]
    reflection = next(event for event in events if event["type"] == "reflection_complete")
    reflection["payload"].update({"status": "complete", "valid_entry_count": 0, "entries": []})
    public_resolution = next(
        event for event in events if event["type"] == "hunter_shot_resolved"
    )
    public_resolution["payload"]["private_reason"] = "must-not-leak"
    events.append(_v2_event(
        "closure-corrupt",
        len(events),
        "action_trace_audit",
        {
            "task_type": "speech",
            "action_trace": {
                "execution_attempts": [
                    {
                        "provider": "same",
                        "model": "same-model",
                        "route_kind": "primary",
                    },
                    {
                        "provider": "same",
                        "model": "same-model",
                        "route_kind": "provider_fallback",
                    },
                ],
            },
        },
    ))
    game["deaths"] = [{
        "player_id": "v1",
        "reason": "wolf_kill",
        "resolution_batch": "night_?",
        "resolution_batch_parse_failed": True,
    }]
    aborted = deepcopy(_closure_game("closure-aborted", single_wolf=True))
    aborted.update({
        "status": "aborted",
        "winning_faction": None,
        "termination_reason": "step_limit",
        "quality_score": {"probe": "saved"},
    })
    aborted_events = aborted["events"]
    assert isinstance(aborted_events, list)
    aborted["events"] = [
        event for event in aborted_events if event["type"] != "game_aborted"
    ]

    report = compute_balance_audit(
        [game, aborted],
        quality_recomputer=lambda source: {"probe": "offline"},
    )

    assert report["malformed_resolution_batch_count"] == 1
    assert report["same_route_provider_fallback_count"] == 1
    assert report["saved_offline_quality_diff_count"] == 2
    assert report["abort_terminal_coverage_rate"] == 0.0
    assert report["majority_wolf_kill_execution_rate"] == 0.0
    assert report["empty_reflection_success_count"] == 1
    assert report["source_event_id_traceability_rate"] < 1.0
    assert report["public_skill_resolution_leak_count"] == 1


def test_balance_audit_fails_closed_when_closure_evidence_is_legacy_or_missing() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    report = compute_balance_audit([{
        "game_id": "legacy-incomplete",
        "winning_faction": "good",
        "players": {"p01": {"role": "villager"}},
        "events": [],
        "deaths": [],
    }])

    assert report["saved_offline_quality_consistency_metrics_supported"] is False
    assert report["provider_fallback_route_metrics_supported"] is False
    assert report["terminal_integrity_metrics_supported"] is False
    assert report["wolf_consensus_execution_metrics_supported"] is False
    assert report["source_event_traceability_metrics_supported"] is False
    assert report["public_exposure_metrics_supported"] is False


def test_eighty_percent_faction_skew_only_requires_more_samples() -> None:
    from scripts.evaluate_audit_closure_thresholds import evaluate_thresholds

    report = _passing_report()
    report.update({"wolf_win_rate": 0.8, "good_win_rate": 0.2})

    result = evaluate_thresholds(report)

    assert result["overall_pass"] is True
    assert result["requires_more_samples"] is True


def test_soak_threshold_requires_exactly_ten_games() -> None:
    from scripts.evaluate_audit_closure_thresholds import evaluate_thresholds

    report = _passing_report()
    report["games"] = 9

    result = evaluate_thresholds(report)
    by_name = {item["name"]: item for item in result["thresholds"]}

    assert by_name["games"] == {
        "name": "games",
        "supported": True,
        "actual": 9,
        "operator": "==",
        "expected": 10,
        "passed": False,
    }
    assert result["overall_pass"] is False


def test_reasoning_thresholds_fail_closed_on_unsupported_critical_attempt() -> None:
    from scripts.evaluate_audit_closure_thresholds import evaluate_thresholds

    report = _passing_report()
    report.update({
        "critical_task_reasoning_request_dispatched_rate": 0.5,
        "critical_task_reasoning_unsupported_count": 1,
    })

    result = evaluate_thresholds(report)
    by_name = {item["name"]: item for item in result["thresholds"]}

    assert by_name["critical_task_reasoning_request_dispatched_rate"]["passed"] is False
    assert by_name["critical_task_reasoning_unsupported_count"]["passed"] is False
    assert result["overall_pass"] is False


def test_semantic_relation_invariants_are_hard_thresholds() -> None:
    from scripts.evaluate_audit_closure_thresholds import evaluate_thresholds

    report = _passing_report()
    report["semantic_repair_speaker_attribution_preservation_rate"] = 0.9
    report["semantic_repair_negation_preservation_rate"] = 0.8

    result = evaluate_thresholds(report)
    by_name = {item["name"]: item for item in result["thresholds"]}

    assert by_name["semantic_repair_speaker_attribution_preservation_rate"][
        "passed"
    ] is False
    assert by_name["semantic_repair_negation_preservation_rate"]["passed"] is False
    assert result["overall_pass"] is False


def test_semantic_observational_rates_do_not_gate_public_evidence_safety() -> None:
    from scripts.evaluate_audit_closure_thresholds import evaluate_thresholds

    report = _passing_report()
    report.update({
        "semantic_repair_target_preservation_rate": 0.0,
        "semantic_repair_no_new_claim_rate": 0.0,
        "semantic_repair_verified_claim_retention_rate": 0.0,
    })

    result = evaluate_thresholds(report)
    by_name = {item["name"]: item for item in result["thresholds"]}

    assert "semantic_repair_target_preservation_rate" not in by_name
    assert "semantic_repair_no_new_claim_rate" not in by_name
    assert "semantic_repair_verified_claim_retention_rate" not in by_name
    assert by_name["semantic_repair_public_evidence_safety_rate"] == {
        "name": "semantic_repair_public_evidence_safety_rate",
        "supported": True,
        "actual": 1.0,
        "operator": "==",
        "expected": 1.0,
        "passed": True,
    }
    assert result["overall_pass"] is True


@pytest.mark.parametrize(
    ("support", "rate"),
    [(False, 1.0), (True, None)],
)
def test_semantic_public_evidence_safety_gate_fails_closed_without_v2_evidence(
    support: object,
    rate: object,
) -> None:
    from scripts.evaluate_audit_closure_thresholds import evaluate_thresholds

    report = _passing_report()
    report.update({
        "semantic_repair_public_evidence_safety_metrics_supported": support,
        "semantic_repair_public_evidence_safety_rate": rate,
    })

    result = evaluate_thresholds(report)
    by_name = {item["name"]: item for item in result["thresholds"]}

    assert by_name["semantic_repair_public_evidence_safety_rate"]["supported"] is False
    assert by_name["semantic_repair_public_evidence_safety_rate"]["actual"] is None
    assert by_name["semantic_repair_public_evidence_safety_rate"]["passed"] is False
    assert result["overall_pass"] is False


def test_dispatched_threshold_does_not_mislabel_unconfirmed_as_confirmed() -> None:
    from scripts.evaluate_audit_closure_thresholds import evaluate_thresholds

    report = _passing_report()
    report.update({
        "critical_task_reasoning_request_dispatched_rate": 1.0,
        "critical_task_reasoning_confirmed_rate": 0.5,
        "critical_task_reasoning_requires_confirmation_review_count": 5,
    })

    result = evaluate_thresholds(report)
    by_name = {item["name"]: item for item in result["thresholds"]}

    assert by_name["critical_task_reasoning_request_dispatched_rate"]["passed"] is True
    assert result["requires_confirmation_review"] is True
    assert result["critical_task_reasoning_confirmed_rate"] == 0.5


def test_real_runtime_semantic_event_flows_through_report_and_threshold() -> None:
    from scripts.evaluate_audit_closure_thresholds import evaluate_thresholds
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics
    from werewolf_agent.evaluation.trace_identity import DecisionIdentity
    from werewolf_agent.runtime.nodes.action_audit import _action_audit_events

    semantic = {
        "repairable": True, "success": True, "target_preserved": True,
        "speaker_attribution_preserved": True, "negation_preserved": True,
        "introduced_claim_count": 0, "verified_claim_count": 1,
        "retained_verified_claim_count": 1, "generic_template_used": False,
        "fallback_kind": "no_fallback",
    }
    identity = DecisionIdentity(
        game_id="g1", player_id="p01", phase="day", day_number=1,
        night_number=0, task_type="speech", action_index=0,
    )
    runtime_events = _action_audit_events(
        state={}, player_id="p01", phase="speech",
        action_trace={"semantic_repair_audit": semantic},
        decision_identity=identity, exposure_collector=None,
    )
    game = {
        "game_id": "g1",
        "players": {"p01": {"role": "villager"}},
        "events": [
        {"type": event.type, "payload": event.payload} for event in runtime_events
    ]}
    metrics = compute_acceptance_audit_metrics([game])
    result = evaluate_thresholds({**_passing_report(), **metrics})
    by_name = {item["name"]: item for item in result["thresholds"]}

    assert by_name["semantic_repair_success_rate"]["passed"] is True
    assert "semantic_repair_target_preservation_rate" not in by_name
    assert "semantic_repair_no_new_claim_rate" not in by_name
    assert "semantic_repair_verified_claim_retention_rate" not in by_name
    assert by_name["semantic_repair_public_evidence_safety_rate"]["supported"] is False
    assert result["overall_pass"] is False


def test_semantic_reconciliation_failures_close_the_public_evidence_threshold() -> None:
    from scripts.evaluate_audit_closure_thresholds import evaluate_thresholds
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    semantic = {
        "repairable": True,
        "semantic_gate_version": 2,
        "success": True,
        "target_preserved": True,
        "speaker_attribution_preserved": True,
        "negation_preserved": True,
        "introduced_claim_count": 0,
        "unsupported_public_claim_count": 0,
        "verified_claim_count": 0,
        "retained_verified_claim_count": 0,
        "generic_template_used": False,
        "fallback_kind": "no_fallback",
    }
    identity = {
        "trace_id": "reconciliation-trace",
        "game_id": "reconciliation-game",
        "action_index": 1,
        "task_type": "speech",
    }
    paired_game = {
        "game_id": "reconciliation-game",
        "players": {"p01": {"role": "villager"}},
        "events": [
            {"type": "semantic_repair_audit", "payload": {**semantic, **identity}},
            {
                "type": "action_trace_audit",
                "payload": {
                    **identity,
                    "action_trace": {"semantic_repair_audit": dict(semantic)},
                },
            },
        ],
    }
    invalid_identity = deepcopy(paired_game)
    invalid_payload = dict(invalid_identity["events"][0]["payload"])
    invalid_payload.pop("trace_id")
    invalid_identity["events"].append({
        "type": "semantic_repair_audit",
        "payload": invalid_payload,
    })
    conflicting_pairs = []
    for nested_updates in (
        {"repairable": False},
        {"target_preserved": False},
        {"introduced_claim_count": 1},
        {"verified_claim_count": 2},
        {"retained_verified_claim_count": 1},
        {"generic_template_used": True},
        {"unsupported_public_claim_count": 1},
        {"rejection_reason_codes": ["unsupported_public_claim"]},
        {"introduced_claim_count": False},
        {"unsupported_public_claim_count": False},
        {"verified_claim_count": False},
        {"retained_verified_claim_count": False},
        {"success": 1},
        {"target_preserved": 1},
        {"speaker_attribution_preserved": 1},
        {"negation_preserved": 1},
        {"generic_template_used": 0},
    ):
        conflicting_pair = deepcopy(paired_game)
        nested = conflicting_pair["events"][1]["payload"]["action_trace"][
            "semantic_repair_audit"
        ]
        nested.update(nested_updates)
        conflicting_pairs.append(conflicting_pair)

    for game in (invalid_identity, *conflicting_pairs):
        metrics = compute_acceptance_audit_metrics([game])
        result = evaluate_thresholds({**_passing_report(), **metrics})

        assert metrics["semantic_repair_metrics_supported"] is False
        assert metrics["semantic_repair_public_evidence_safety_metrics_supported"] is False
        assert metrics["semantic_repair_public_evidence_safety_rate"] is None
        assert result["overall_pass"] is False


def test_cli_writes_threshold_file_before_returning_failure(tmp_path) -> None:
    from scripts.evaluate_audit_closure_thresholds import main

    report_path = tmp_path / "report.json"
    output_path = tmp_path / "audit-closure-thresholds.json"
    report = _passing_report()
    report.pop("completion_rate")
    report_path.write_text(json.dumps(report), encoding="utf-8")

    exit_code = main([str(report_path), str(output_path)])

    assert exit_code == 1
    written = json.loads(output_path.read_text(encoding="utf-8-sig"))
    assert written["overall_pass"] is False
    assert next(
        item for item in written["thresholds"]
        if item["name"] == "completion_rate"
    )["supported"] is False
