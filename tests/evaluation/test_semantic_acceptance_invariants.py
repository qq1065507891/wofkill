# -*- coding: utf-8 -*-
"""
验证 V1/V2 语义修复验收指标与公开证据安全门控。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-19
"""

from __future__ import annotations


def _game(
    *,
    speaker_preserved: object,
    negation_preserved: object,
    game_id: str = "g1",
    action_index: int = 1,
    **semantic_overrides: object,
) -> dict[str, object]:
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
        **semantic_overrides,
    }
    identity = {
        "trace_id": "trace-1",
        "game_id": game_id,
        "action_index": action_index,
        "task_type": "speech",
    }
    return {
        "game_id": game_id,
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


def test_v1_semantic_success_calculation_remains_unchanged_without_version() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    metrics = compute_acceptance_audit_metrics([
        _game(
            speaker_preserved=True,
            negation_preserved=True,
            target_preserved=False,
            introduced_claim_count=1,
            retained_verified_claim_count=0,
        ),
    ])

    assert metrics["semantic_repair_success_count"] == 0
    assert metrics["semantic_repair_success_rate"] == 0.0
    assert metrics["semantic_repair_public_evidence_safety_metrics_supported"] is False
    assert metrics["semantic_repair_public_evidence_safety_rate"] is None


def test_v2_success_uses_public_evidence_safety_not_observational_metrics() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    metrics = compute_acceptance_audit_metrics([
        _game(
            speaker_preserved=True,
            negation_preserved=True,
            semantic_gate_version=2,
            target_preserved=False,
            introduced_claim_count=1,
            verified_claim_count=2,
            retained_verified_claim_count=1,
            unsupported_public_claim_count=0,
        ),
    ])

    assert metrics["semantic_repair_success_count"] == 1
    assert metrics["semantic_repair_success_rate"] == 1.0
    assert metrics["semantic_repair_target_preservation_rate"] == 0.0
    assert metrics["semantic_repair_no_new_claim_rate"] == 0.0
    assert metrics["semantic_repair_verified_claim_retention_rate"] == 0.0
    assert metrics["semantic_repair_public_evidence_safety_metrics_supported"] is True
    assert metrics["semantic_repair_public_evidence_safety_rate"] == 1.0


def test_v2_public_evidence_safety_rate_counts_unsafe_rows() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    metrics = compute_acceptance_audit_metrics([
        _game(
            speaker_preserved=True,
            negation_preserved=True,
            semantic_gate_version=2,
            success=False,
            generic_template_used=True,
            fallback_kind="generic_template",
            unsupported_public_claim_count=1,
        ),
    ])

    assert metrics["semantic_repair_public_evidence_safety_metrics_supported"] is True
    assert metrics["semantic_repair_public_evidence_safety_rate"] == 0.0


def test_v2_boolean_unsupported_public_claim_count_is_not_success() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    metrics = compute_acceptance_audit_metrics([
        _game(
            speaker_preserved=True,
            negation_preserved=True,
            semantic_gate_version=2,
            unsupported_public_claim_count=False,
        ),
    ])

    assert metrics["semantic_repair_success_count"] == 0
    assert metrics["semantic_repair_public_evidence_safety_metrics_supported"] is False
    assert metrics["semantic_repair_public_evidence_safety_rate"] is None


def test_public_evidence_safety_fails_closed_for_mixed_or_incomplete_v2_rows() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    mixed = compute_acceptance_audit_metrics([
        _game(speaker_preserved=True, negation_preserved=True),
        _game(
            speaker_preserved=True,
            negation_preserved=True,
            game_id="g2",
            semantic_gate_version=2,
            unsupported_public_claim_count=0,
        ),
    ])
    missing_v2_count = compute_acceptance_audit_metrics([
        _game(
            speaker_preserved=True,
            negation_preserved=True,
            semantic_gate_version=2,
        ),
    ])
    malformed_v2_count = compute_acceptance_audit_metrics([
        _game(
            speaker_preserved=True,
            negation_preserved=True,
            semantic_gate_version=2,
            unsupported_public_claim_count=-1,
        ),
    ])
    string_v2_count = compute_acceptance_audit_metrics([
        _game(
            speaker_preserved=True,
            negation_preserved=True,
            semantic_gate_version=2,
            unsupported_public_claim_count="0",
        ),
    ])

    assert mixed["semantic_repair_public_evidence_safety_metrics_supported"] is False
    assert mixed["semantic_repair_public_evidence_safety_rate"] is None
    assert (
        missing_v2_count["semantic_repair_public_evidence_safety_metrics_supported"]
        is False
    )
    assert missing_v2_count["semantic_repair_success_count"] == 0
    assert missing_v2_count["semantic_repair_public_evidence_safety_rate"] is None
    assert malformed_v2_count["semantic_repair_success_count"] == 0
    assert malformed_v2_count[
        "semantic_repair_public_evidence_safety_metrics_supported"
    ] is False
    assert malformed_v2_count["semantic_repair_public_evidence_safety_rate"] is None
    assert string_v2_count["semantic_repair_success_count"] == 0
    assert string_v2_count["semantic_repair_public_evidence_safety_metrics_supported"] is False
    assert string_v2_count["semantic_repair_public_evidence_safety_rate"] is None


def test_invalid_standalone_semantic_audit_identity_fails_reconciliation_closed() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    game = _game(
        speaker_preserved=True,
        negation_preserved=True,
        semantic_gate_version=2,
        unsupported_public_claim_count=0,
    )
    invalid_standalone = dict(game["events"][0]["payload"])
    invalid_standalone.pop("trace_id")
    game["events"].append({
        "type": "semantic_repair_audit",
        "payload": invalid_standalone,
    })

    metrics = compute_acceptance_audit_metrics([game])

    assert metrics["semantic_repair_metrics_supported"] is False
    assert metrics["semantic_repair_success_rate"] is None
    assert metrics["semantic_repair_public_evidence_safety_metrics_supported"] is False
    assert metrics["semantic_repair_public_evidence_safety_rate"] is None


def test_conflicting_paired_v2_semantic_audits_fail_reconciliation_closed() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    game = _game(
        speaker_preserved=True,
        negation_preserved=True,
        semantic_gate_version=2,
        unsupported_public_claim_count=0,
    )
    nested = game["events"][1]["payload"]["action_trace"][
        "semantic_repair_audit"
    ]
    nested["unsupported_public_claim_count"] = 1

    metrics = compute_acceptance_audit_metrics([game])

    assert metrics["semantic_repair_metrics_supported"] is False
    assert metrics["semantic_repair_success_rate"] is None
    assert metrics["semantic_repair_public_evidence_safety_metrics_supported"] is False
    assert metrics["semantic_repair_public_evidence_safety_rate"] is None


def test_paired_semantic_audit_reconciliation_compares_every_decisive_field() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    for nested_updates in (
        {"repairable": False},
        {"semantic_gate_version": 1},
        {"success": False},
        {"target_preserved": False},
        {"speaker_attribution_preserved": False},
        {"negation_preserved": False},
        {"introduced_claim_count": 1},
        {"verified_claim_count": 2},
        {"retained_verified_claim_count": 0},
        {"generic_template_used": True},
        {"fallback_kind": "task_specific"},
        {"rejection_reason_codes": ["unsupported_public_claim"]},
    ):
        game = _game(
            speaker_preserved=True,
            negation_preserved=True,
            semantic_gate_version=2,
            unsupported_public_claim_count=0,
        )
        nested = game["events"][1]["payload"]["action_trace"][
            "semantic_repair_audit"
        ]
        nested.update(nested_updates)

        metrics = compute_acceptance_audit_metrics([game])

        assert metrics["semantic_repair_metrics_supported"] is False
        assert metrics["semantic_repair_public_evidence_safety_metrics_supported"] is False
        assert metrics["semantic_repair_public_evidence_safety_rate"] is None


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
        "attempt_count": 0,
        "retry_count": 0,
        "provider_fallback_count": 0,
        "execution_attempts": [],
        "final_action": {"action_type": "speech", "target_id": None, "reason": "fallback"},
    })
    game["events"].append({
        "type": "wolf_team_plan_fallback",
        "payload": {
            "generated_by": "terminal_fallback",
            "decision_outcome": "terminal_fallback",
            "terminal_failure_code": "empty_response",
            "original_failure_code": "empty_response",
            "failure_stage": "model_output",
            "fallback_kind": "wolf_team_plan_structured_stance",
            "attempts": 0,
            "final_action": {"action_type": "wolf_team_plan", "target_id": None, "reason": "fallback"},
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
    assert metrics["terminal_fallback_original_failure_code_metrics_supported"] is False
    assert metrics["terminal_fallback_original_failure_code_coverage_rate"] is None


def test_terminal_failure_coverage_rejects_uninformative_original_codes() -> None:
    from werewolf_agent.evaluation.acceptance_audit import (
        compute_acceptance_audit_metrics,
    )

    games = []
    for index, original_code in enumerate((None, "", "unknown"), 1):
        game = _game(speaker_preserved=True, negation_preserved=True)
        game["game_id"] = f"g-uninformative-{index}"
        for event in game["events"]:
            event["payload"]["game_id"] = game["game_id"]
        trace = game["events"][1]["payload"]["action_trace"]
        trace.update({
            "generated_by": "terminal_fallback",
            "decision_outcome": "terminal_fallback",
            "terminal_failure_code": original_code or "unknown",
            "original_failure_code": original_code,
            "failure_stage": "protocol",
            "fallback_kind": "ordinary_speech",
        })
        games.append(game)

    metrics = compute_acceptance_audit_metrics(games)

    assert metrics["terminal_fallback_count"] == 3
    assert metrics["terminal_fallback_original_failure_code_covered_count"] == 0
    assert metrics["terminal_fallback_original_failure_code_metrics_supported"] is False
    assert metrics["terminal_fallback_original_failure_code_coverage_rate"] is None


def test_terminal_metrics_reject_spoofed_or_inconsistent_v2_rows() -> None:
    from werewolf_agent.evaluation.acceptance_audit import compute_acceptance_audit_metrics

    game = _game(speaker_preserved=True, negation_preserved=True)
    trace = game["events"][1]["payload"]["action_trace"]
    trace.update({
        "generated_by": "model",
        "decision_outcome": "terminal_fallback",
        "terminal_failure_code": "schema_validation",
        "original_failure_code": "schema_validation",
        "failure_stage": "made_up_stage",
        "fallback_kind": "invented_kind",
        "attempt_count": 1,
        "execution_attempts": [],
        "final_action": {"action_type": "speech", "target_id": None, "reason": "fallback"},
    })

    metrics = compute_acceptance_audit_metrics([game])

    assert metrics["terminal_fallback_count"] == 1
    assert metrics["terminal_fallback_original_failure_code_metrics_supported"] is False
    assert metrics["terminal_fallback_original_failure_code_coverage_rate"] is None
