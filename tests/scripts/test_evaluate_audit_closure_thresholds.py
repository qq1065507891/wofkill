# -*- coding: utf-8 -*-
"""
验证审查闭环浸泡测试的硬门槛投影与失败落盘顺序。

作者: Project contributors
创建日期: 2026-07-13
"""

from __future__ import annotations

import json


def _passing_report() -> dict[str, object]:
    return {
        "games": 10,
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
        "critical_task_reasoning_status_metrics_supported": True,
        "critical_task_reasoning_status_explicit_rate": 1.0,
        "reasoning_fallback_disabled_count": 0,
        "unsupported_public_fact_claim_count": 0,
        "reflection_contamination_metrics_supported": True,
        "reflection_persisted_rejected_fact_count": 0,
        "semantic_repair_metrics_supported": True,
        "semantic_repair_target_preservation_rate": 1.0,
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
    assert len(result["thresholds"]) == 22
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
