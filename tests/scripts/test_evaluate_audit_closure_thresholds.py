# -*- coding: utf-8 -*-
"""
验证审查闭环浸泡测试的硬门槛投影与失败落盘顺序。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-15
"""

from __future__ import annotations

import json


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
        "possible_world_metrics_supported": True,
        "possible_world_unique_rate": 1.0,
        "possible_world_evidence_coverage_rate": 1.0,
        "power_role_evidence_metrics_supported": True,
        "power_role_evidence_completeness_rate": 1.0,
        "decision_execution_metrics_supported": True,
        "attempt_retry_consistency_error_count": 0,
        "wolf_win_rate": 0.5,
        "good_win_rate": 0.5,
    }


def test_all_hard_thresholds_pass_with_complete_descriptors() -> None:
    from scripts.evaluate_audit_closure_thresholds import evaluate_thresholds

    result = evaluate_thresholds(_passing_report())

    assert result["overall_pass"] is True
    assert result["requires_more_samples"] is False
    assert len(result["thresholds"]) == 34
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
    assert by_name["semantic_repair_target_preservation_rate"]["passed"] is True
    assert by_name["semantic_repair_no_new_claim_rate"]["passed"] is True
    assert by_name["semantic_repair_verified_claim_retention_rate"]["passed"] is True


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
