# -*- coding: utf-8 -*-
"""
验证语义修复说话者归属与否定关系进入最终验收指标。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-16
"""

from __future__ import annotations


def _game(*, speaker_preserved: object, negation_preserved: object) -> dict[str, object]:
    semantic = {
        "repairable": True,
        "success": True,
        "target_preserved": True,
        "speaker_attribution_preserved": speaker_preserved,
        "negation_preserved": negation_preserved,
        "introduced_claim_count": 0,
        "verified_claim_count": 1,
        "retained_verified_claim_count": 1,
        "generic_template_used": False,
        "fallback_kind": "no_fallback",
    }
    identity = {
        "trace_id": "trace-1",
        "game_id": "g1",
        "action_index": 1,
        "task_type": "speech",
    }
    return {
        "game_id": "g1",
        "players": {"p01": {"role": "villager"}},
        "events": [
            {
                "type": "semantic_repair_audit",
                "payload": {**semantic, **identity},
            },
            {
                "type": "action_trace_audit",
                "payload": {
                    **identity,
                    "action_trace": {"semantic_repair_audit": semantic},
                },
            },
        ],
    }


def test_acceptance_reports_speaker_and_negation_preservation_rates() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    metrics = compute_acceptance_audit_metrics([
        _game(speaker_preserved=True, negation_preserved=True),
    ])

    assert metrics["semantic_repair_metrics_supported"] is True
    assert metrics["semantic_repair_speaker_attribution_preservation_rate"] == 1.0
    assert metrics["semantic_repair_negation_preservation_rate"] == 1.0


def test_acceptance_fails_closed_when_semantic_invariant_is_missing() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    metrics = compute_acceptance_audit_metrics([
        _game(speaker_preserved=None, negation_preserved=True),
    ])

    assert metrics["semantic_repair_metrics_supported"] is False
    assert metrics["semantic_repair_speaker_attribution_preservation_rate"] is None
    assert metrics["semantic_repair_negation_preservation_rate"] is None


def test_terminal_failure_coverage_is_independent_from_semantic_acceptance() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    game = _game(speaker_preserved=True, negation_preserved=True)
    trace_event = game["events"][1]
    trace = trace_event["payload"]["action_trace"]
    trace.update({
        "generated_by": "terminal_fallback",
        "decision_outcome": "terminal_fallback",
        "terminal_failure_code": "schema_validation",
        "original_failure_code": "schema_validation",
        "failure_stage": "protocol",
        "fallback_kind": "ordinary_speech",
    })
    game["events"].append({
        "type": "wolf_team_plan_fallback",
        "payload": {
            "generated_by": "terminal_fallback",
            "terminal_failure_code": "empty_response",
            "original_failure_code": "empty_response",
            "failure_stage": "model_output",
            "fallback_kind": "wolf_team_plan_structured_stance",
        },
    })

    metrics = compute_acceptance_audit_metrics([game])

    assert metrics["semantic_repair_success_rate"] == 1.0
    assert metrics["terminal_fallback_count"] == 2
    assert metrics["terminal_fallback_original_failure_code_covered_count"] == 2
    assert metrics["terminal_fallback_original_failure_code_coverage_rate"] == 1.0
    assert metrics["terminal_fallback_kind_counts"] == {
        "ordinary_speech": 1,
        "wolf_team_plan_structured_stance": 1,
    }


def test_terminal_failure_coverage_fails_closed_when_required_field_is_missing() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    game = _game(speaker_preserved=True, negation_preserved=True)
    trace = game["events"][1]["payload"]["action_trace"]
    trace.update({
        "generated_by": "terminal_fallback",
        "decision_outcome": "terminal_fallback",
        "terminal_failure_code": "schema_validation",
        "failure_stage": "protocol",
        "fallback_kind": "ordinary_speech",
    })

    metrics = compute_acceptance_audit_metrics([game])

    assert metrics["terminal_fallback_count"] == 1
    assert metrics["terminal_fallback_original_failure_code_covered_count"] == 0
    assert metrics["terminal_fallback_original_failure_code_coverage_rate"] == 0.0
