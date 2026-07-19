# -*- coding: utf-8 -*-
"""
把十局审计报告投影为逐项硬门槛结果，并在失败时保持可诊断产物。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-19

使用示例:
    python scripts/evaluate_audit_closure_thresholds.py report.json thresholds.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


_THRESHOLDS = (
    ("games", "==", 10, "games_present"),
    ("unique_game_id_count", "==", 10, "games"),
    ("unique_game_artifact_path_count", "==", 10, "games"),
    ("completion_rate", "==", 1.0, "games"),
    ("terminal_post_win_game_model_call_count", "==", 0, "games"),
    ("weak_wolf_plan_kill_count", "==", 0, "games"),
    ("fallback_plan_kill_without_target_evidence_count", "==", 0, "games"),
    ("wolf_team_plan_terminal_fallback_rate", "<", 0.10, "wolf_team_plan_outcome_metrics_supported"),
    ("wolf_plan_schema_fallback_rate", "<", 0.05, "wolf_team_plan_outcome_metrics_supported"),
    ("persona_injection_confirmation_rate", "==", 1.0, "persona_injection_confirmation_metrics_supported"),
    ("persona_exposure_linkage_rate", "==", 1.0, "persona_exposure_linkage_metrics_supported"),
    ("critical_task_reasoning_request_coverage", "==", 1.0, "critical_task_reasoning_request_coverage_supported"),
    ("critical_task_reasoning_request_dispatched_rate", "==", 1.0, "critical_task_reasoning_request_dispatched_metrics_supported"),
    ("critical_task_reasoning_unsupported_count", "==", 0, "critical_task_reasoning_request_dispatched_metrics_supported"),
    ("critical_task_reasoning_not_requested_count", "==", 0, "critical_task_reasoning_request_dispatched_metrics_supported"),
    ("critical_task_reasoning_fallback_disabled_count", "==", 0, "critical_task_reasoning_request_dispatched_metrics_supported"),
    ("critical_task_reasoning_status_explicit_rate", "==", 1.0, "critical_task_reasoning_status_metrics_supported"),
    ("reasoning_task_type_missing_count", "==", 0, "games"),
    ("reasoning_task_type_unknown_count", "==", 0, "games"),
    ("decision_execution_invalid_sequence_count", "==", 0, "games"),
    ("reasoning_fallback_disabled_count", "==", 0, "decision_execution_metrics_supported"),
    ("unsupported_public_fact_claim_count", "==", 0, "games"),
    ("reflection_persisted_rejected_fact_count", "==", 0, "reflection_contamination_metrics_supported"),
    ("semantic_repair_speaker_attribution_preservation_rate", "==", 1.0, "semantic_repair_metrics_supported"),
    ("semantic_repair_negation_preservation_rate", "==", 1.0, "semantic_repair_metrics_supported"),
    ("semantic_repair_success_rate", ">=", 0.95, "semantic_repair_metrics_supported"),
    ("semantic_repair_generic_template_count", "==", 0, "semantic_repair_metrics_supported"),
    ("semantic_repair_public_evidence_safety_rate", "==", 1.0, "semantic_repair_public_evidence_safety_metrics_supported"),
    ("possible_world_unique_rate", "==", 1.0, "possible_world_metrics_supported"),
    ("possible_world_evidence_coverage_rate", "==", 1.0, "possible_world_metrics_supported"),
    ("power_role_evidence_completeness_rate", "==", 1.0, "power_role_evidence_metrics_supported"),
    ("attempt_retry_consistency_error_count", "==", 0, "decision_execution_metrics_supported"),
    ("malformed_resolution_batch_count", "==", 0, "resolution_batch_integrity_metrics_supported"),
    ("same_route_provider_fallback_count", "==", 0, "provider_fallback_route_metrics_supported"),
    ("saved_offline_quality_diff_count", "==", 0, "saved_offline_quality_consistency_metrics_supported"),
    ("finished_without_winner_count", "==", 0, "terminal_integrity_metrics_supported"),
    ("abort_terminal_coverage_rate", "==", 1.0, "abort_terminal_coverage_metrics_supported"),
    ("majority_wolf_kill_execution_rate", "==", 1.0, "majority_wolf_kill_execution_metrics_supported"),
    ("single_wolf_kill_execution_rate", "==", 1.0, "single_wolf_kill_execution_metrics_supported"),
    ("empty_reflection_success_count", "==", 0, "reflection_transaction_metrics_supported"),
    ("source_event_id_traceability_rate", "==", 1.0, "source_event_traceability_metrics_supported"),
    ("public_skill_resolution_leak_count", "==", 0, "public_exposure_metrics_supported"),
)


def _is_supported(report: dict[str, Any], name: str, support_key: str) -> bool:
    if name not in report or report[name] is None:
        return False
    if support_key == "games_present":
        value = report.get(name)
        return isinstance(value, int) and not isinstance(value, bool)
    support = report.get(support_key)
    if support_key == "games":
        return isinstance(support, int) and not isinstance(support, bool) and support > 0
    return support is True


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return False
    if operator == "==":
        return actual == expected
    if operator == "<":
        return actual < expected
    if operator == ">=":
        return actual >= expected
    raise ValueError(f"unsupported threshold operator: {operator}")


def evaluate_thresholds(report: dict[str, Any]) -> dict[str, Any]:
    """逐项 fail closed 评估硬门槛；胜率倾斜只触发扩大样本提示。"""
    thresholds: list[dict[str, Any]] = []
    for name, operator, expected, support_key in _THRESHOLDS:
        supported = _is_supported(report, name, support_key)
        actual = report.get(name) if supported else None
        thresholds.append({
            "name": name,
            "supported": supported,
            "actual": actual,
            "operator": operator,
            "expected": expected,
            "passed": supported and _compare(actual, operator, expected),
        })
    faction_rates = (report.get("wolf_win_rate"), report.get("good_win_rate"))
    requires_more_samples = any(
        isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and rate >= 0.8
        for rate in faction_rates
    )
    confirmed_rate = report.get("critical_task_reasoning_confirmed_rate")
    confirmation_review_count = report.get(
        "critical_task_reasoning_requires_confirmation_review_count"
    )
    requires_confirmation_review = (
        not isinstance(confirmed_rate, (int, float))
        or isinstance(confirmed_rate, bool)
        or confirmed_rate < 1.0
        or not isinstance(confirmation_review_count, int)
        or isinstance(confirmation_review_count, bool)
        or confirmation_review_count > 0
    )
    def soak_count(name: str) -> int | None:
        value = report.get(name)
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else None
        )

    return {
        "overall_pass": all(item["passed"] for item in thresholds),
        "requires_more_samples": requires_more_samples,
        "requires_confirmation_review": requires_confirmation_review,
        "critical_task_reasoning_confirmed_rate": (
            confirmed_rate
            if isinstance(confirmed_rate, (int, float))
            and not isinstance(confirmed_rate, bool)
            else None
        ),
        "soak_summary": {
            "launch_count": soak_count("soak_launch_count"),
            "finished_count": soak_count("soak_finished_count"),
            "aborted_count": soak_count("soak_aborted_count"),
        },
        "thresholds": thresholds,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """评估并始终先写结果文件，再用退出码表达硬门槛是否通过。"""
    parser = argparse.ArgumentParser(description="Evaluate audit closure thresholds")
    parser.add_argument("report_json", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.report_json.read_text(encoding="utf-8-sig"))
    result = evaluate_thresholds(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
