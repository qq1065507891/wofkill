# -*- coding: utf-8 -*-
"""
把十局审计报告投影为逐项硬门槛结果，并在失败时保持可诊断产物。

作者: Project contributors
创建日期: 2026-07-13

使用示例:
    python scripts/evaluate_audit_closure_thresholds.py report.json thresholds.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


_THRESHOLDS = (
    ("completion_rate", "==", 1.0, "games"),
    ("terminal_post_win_game_model_call_count", "==", 0, "games"),
    ("weak_wolf_plan_kill_count", "==", 0, "games"),
    ("fallback_plan_kill_without_target_evidence_count", "==", 0, "games"),
    ("wolf_team_plan_terminal_fallback_rate", "<", 0.10, "wolf_team_plan_outcome_metrics_supported"),
    ("wolf_plan_schema_fallback_rate", "<", 0.05, "wolf_team_plan_outcome_metrics_supported"),
    ("persona_injection_confirmation_rate", "==", 1.0, "persona_injection_confirmation_metrics_supported"),
    ("persona_exposure_linkage_rate", "==", 1.0, "persona_exposure_linkage_metrics_supported"),
    ("critical_task_reasoning_request_coverage", "==", 1.0, "critical_task_reasoning_request_coverage_supported"),
    ("critical_task_reasoning_status_explicit_rate", "==", 1.0, "critical_task_reasoning_status_metrics_supported"),
    ("reasoning_fallback_disabled_count", "==", 0, "decision_execution_metrics_supported"),
    ("unsupported_public_fact_claim_count", "==", 0, "games"),
    ("reflection_persisted_rejected_fact_count", "==", 0, "reflection_contamination_metrics_supported"),
    ("semantic_repair_target_preservation_rate", "==", 1.0, "semantic_repair_metrics_supported"),
    ("semantic_repair_no_new_claim_rate", "==", 1.0, "semantic_repair_metrics_supported"),
    ("semantic_repair_success_rate", ">=", 0.95, "semantic_repair_metrics_supported"),
    ("semantic_repair_generic_template_count", "==", 0, "semantic_repair_metrics_supported"),
    ("possible_world_unique_rate", "==", 1.0, "possible_world_metrics_supported"),
    ("possible_world_evidence_coverage_rate", "==", 1.0, "possible_world_metrics_supported"),
    ("power_role_evidence_completeness_rate", "==", 1.0, "power_role_evidence_metrics_supported"),
    ("attempt_retry_consistency_error_count", "==", 0, "decision_execution_metrics_supported"),
    ("semantic_repair_verified_claim_retention_rate", "==", 1.0, "semantic_repair_verified_claim_retention_metrics_supported"),
)


def _is_supported(report: dict[str, Any], name: str, support_key: str) -> bool:
    if name not in report or report[name] is None:
        return False
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
    return {
        "overall_pass": all(item["passed"] for item in thresholds),
        "requires_more_samples": requires_more_samples,
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
