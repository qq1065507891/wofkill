# -*- coding: utf-8 -*-
"""
验证验收报告只消费强类型执行事实，并为所有比率保留精确分母。

作者: Project contributors
创建日期: 2026-07-13
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from werewolf_agent.model_gateway.execution_records import (
    AttemptExecutionRecord,
    AttemptOutcome,
    EvidenceKind,
    OpaqueRequestId,
    ReasoningLevel,
    ReasoningStatus,
    RootCause,
    RouteKind,
)


def _attempt(
    ordinal: int,
    route: RouteKind,
    outcome: AttemptOutcome,
    *,
    cause: RootCause = RootCause.NONE,
    provider: str = "primary",
    reasoning_status: ReasoningStatus = ReasoningStatus.CONFIRMED,
) -> AttemptExecutionRecord:
    evidence = {
        ReasoningStatus.CONFIRMED: EvidenceKind.TOKEN_COUNT,
        ReasoningStatus.REQUESTED_UNCONFIRMED: EvidenceKind.NONE,
        ReasoningStatus.UNSUPPORTED: EvidenceKind.UNSUPPORTED,
        ReasoningStatus.FALLBACK_DISABLED: EvidenceKind.FALLBACK_DISABLED,
    }[reasoning_status]
    return AttemptExecutionRecord(
        opaque_request_id=OpaqueRequestId.new("game", "aabbccdd"),
        ordinal=ordinal,
        provider=provider,
        model="model-a",
        route_kind=route,
        root_cause=cause,
        attempt_outcome=outcome,
        requested_reasoning_level=ReasoningLevel.HIGH,
        normalized_reasoning_status=reasoning_status,
        reasoning_token_count=1 if reasoning_status is ReasoningStatus.CONFIRMED else 0,
        evidence_kind=evidence,
    )


@pytest.mark.parametrize(
    ("attempts", "decision", "root_counts", "attempt_counts", "fallback_kept"),
    [
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT),
                _attempt(2, RouteKind.RETRY, AttemptOutcome.SUCCESS),
            ),
            "retry_success",
            {"timeout": 1, "none": 1},
            {"attempt_failure": 1, "attempt_success": 1},
            None,
        ),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.PROVIDER_ERROR),
                _attempt(
                    2,
                    RouteKind.PROVIDER_FALLBACK,
                    AttemptOutcome.SUCCESS,
                    provider="backup",
                ),
            ),
            "provider_fallback_success",
            {"provider_error": 1, "none": 1},
            {"attempt_failure": 1, "attempt_success": 1},
            1,
        ),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.INVALID_OUTPUT),
                _attempt(2, RouteKind.REPAIR, AttemptOutcome.SUCCESS),
            ),
            "repaired_success",
            {"invalid_output": 1, "none": 1},
            {"attempt_failure": 1, "attempt_success": 1},
            None,
        ),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.INVALID_OUTPUT),
                _attempt(2, RouteKind.REPAIR, AttemptOutcome.SUCCESS),
            ),
            "repaired_success",
            {"invalid_output": 1, "none": 1},
            {"attempt_failure": 1, "attempt_success": 1},
            None,
        ),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.POLICY_REJECTION),
                _attempt(
                    2,
                    RouteKind.SAFE_FALLBACK,
                    AttemptOutcome.FAILURE,
                    cause=RootCause.POLICY_REJECTION,
                ),
            ),
            "terminal_fallback",
            {"policy_rejection": 2},
            {"attempt_failure": 2},
            None,
        ),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.INVALID_OUTPUT),
                _attempt(
                    2,
                    RouteKind.SAFE_FALLBACK,
                    AttemptOutcome.FAILURE,
                    cause=RootCause.INVALID_OUTPUT,
                ),
            ),
            "terminal_fallback",
            {"invalid_output": 2},
            {"attempt_failure": 2},
            None,
        ),
    ],
    ids=[
        "provider-timeout-retry-success",
        "provider-fallback-reasoning-preserved",
        "schema-normalization",
        "semantic-repair",
        "terminal-safe-fallback",
        "illegal-action",
    ],
)
def test_execution_report_taxonomy_is_consistent(
    attempts,
    decision,
    root_counts,
    attempt_counts,
    fallback_kept,
) -> None:
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    trace = {
        "execution_attempts": attempts,
        # 故意放入错误的历史投影，证明报告重新消费 translator，而不是信任自由文本。
        "decision_outcome": "legacy_free_text",
        "retry_count": len(attempts) - 1,
        "total_retry_count_until_success": (
            None if decision == "terminal_fallback" else len(attempts) - 1
        ),
    }
    metrics = compute_decision_execution_metrics([{
        "events": [{"type": "action_trace_audit", "payload": {
            "task_type": "vote",
            "action_trace": trace,
        }}],
    }])

    assert metrics["decision_count"] == 1
    assert metrics["attempt_count"] == len(attempts)
    assert metrics["retry_count"] == len(attempts) - 1
    assert metrics["root_cause_counts"] == root_counts
    assert metrics["attempt_outcome_counts"] == attempt_counts
    assert metrics["decision_outcome_counts"] == {decision: 1}
    assert metrics["attempt_retry_consistency_error_count"] == 0
    assert metrics["reasoning_requested_count"] == len(attempts)
    assert metrics["reasoning_confirmed_count"] == len(attempts)
    assert metrics["reasoning_confirmation_rate"] == 1.0
    assert metrics["reasoning_confirmation_supported"] is True
    if fallback_kept is None:
        assert metrics["reasoning_fallback_keep_metrics_supported"] is False
        assert metrics["reasoning_fallback_keep_rate"] is None
    else:
        assert metrics["reasoning_fallback_keep_metrics_supported"] is True
        assert metrics["reasoning_fallback_keep_count"] == fallback_kept
        assert metrics["reasoning_fallback_keep_rate"] == 1.0


def test_execution_report_accepts_json_normalized_attempts() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    attempts = (
        _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT),
        _attempt(2, RouteKind.RETRY, AttemptOutcome.SUCCESS),
    )
    serialized = []
    for item in attempts:
        serialized.append({
            "opaque_request_id": {"value": item.opaque_request_id.value},
            "ordinal": item.ordinal,
            "provider": item.provider,
            "model": item.model,
            "route_kind": item.route_kind.value,
            "root_cause": item.root_cause.value,
            "attempt_outcome": item.attempt_outcome.value,
            "requested_reasoning_level": item.requested_reasoning_level.value,
            "normalized_reasoning_status": item.normalized_reasoning_status.value,
            "reasoning_token_count": item.reasoning_token_count,
            "evidence_kind": item.evidence_kind.value,
        })
    metrics = compute_decision_execution_metrics([{
        "events": [{"type": "action_trace_audit", "payload": {
            "action_trace": {"execution_attempts": serialized},
        }}],
    }])

    assert metrics["decision_outcome_counts"] == {"retry_success": 1}
    assert metrics["attempt_count"] == 2
    assert metrics["retry_count"] == 1


def test_retry_consistency_accepts_two_attempts_with_one_retry_and_no_errors() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    attempts = (
        _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT),
        _attempt(2, RouteKind.RETRY, AttemptOutcome.SUCCESS),
    )
    metrics = compute_decision_execution_metrics([{
        "events": [{"type": "action_trace_audit", "payload": {
            "task_type": "vote",
            "action_trace": {
                "execution_attempts": attempts,
                "retry_count": 1,
                "total_retry_count_until_success": 1,
            },
        }}],
    }])

    assert metrics["attempt_retry_consistency_error_count"] == 0


def test_critical_reasoning_coverage_uses_policy_minimum_and_requires_task_type() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    medium_attempt = replace(_attempt(
        1,
        RouteKind.PRIMARY,
        AttemptOutcome.SUCCESS,
        reasoning_status=ReasoningStatus.REQUESTED_UNCONFIRMED,
    ),
        requested_reasoning_level=ReasoningLevel.MEDIUM,
    )
    none_attempt = replace(
        medium_attempt,
        requested_reasoning_level=ReasoningLevel.NONE,
        normalized_reasoning_status=ReasoningStatus.NOT_REQUESTED,
        evidence_kind=EvidenceKind.NONE,
    )
    games = [{"events": [
        {"type": "action_trace_audit", "payload": {
            "task_type": "wolf_team_plan",
            "action_trace": {"execution_attempts": (medium_attempt,)},
        }},
        {"type": "action_trace_audit", "payload": {
            "task_type": "speech",
            "action_trace": {"execution_attempts": (medium_attempt,)},
        }},
        {"type": "action_trace_audit", "payload": {
            "task_type": "judge_phase",
            "action_trace": {"execution_attempts": (none_attempt,)},
        }},
        {"type": "action_trace_audit", "payload": {
            "action_trace": {"execution_attempts": (medium_attempt,)},
        }},
    ]}]

    metrics = compute_decision_execution_metrics(games)

    assert metrics["critical_task_reasoning_request_count"] == 2
    assert metrics["critical_task_reasoning_requested_count"] == 1
    assert metrics["critical_task_reasoning_request_coverage"] == 0.5
    assert metrics["reasoning_confirmation_supported"] is True
    assert metrics["reasoning_confirmation_rate"] == 0.0
    assert metrics["reasoning_task_type_missing_count"] == 1


def test_none_only_reasoning_has_independent_unsupported_confirmation_rate() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    attempt = replace(
        _attempt(1, RouteKind.PRIMARY, AttemptOutcome.SUCCESS),
        requested_reasoning_level=ReasoningLevel.NONE,
        normalized_reasoning_status=ReasoningStatus.NOT_REQUESTED,
        reasoning_token_count=0,
        evidence_kind=EvidenceKind.NONE,
    )
    metrics = compute_decision_execution_metrics([{"events": [{
        "type": "action_trace_audit",
        "payload": {
            "task_type": "judge_phase",
            "action_trace": {"execution_attempts": (attempt,)},
        },
    }]}])

    assert metrics["reasoning_confirmation_supported"] is False
    assert metrics["reasoning_confirmation_rate"] is None
    assert metrics["critical_task_reasoning_request_coverage_supported"] is False


def test_acceptance_metrics_use_exact_denominators_and_explicit_support() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    report = compute_balance_audit([{
        "game_id": "g1",
        "winning_faction": "good",
        "events": [
            {"type": "victory", "payload": {"winner": "good"}},
            {"type": "action_trace_audit", "payload": {"task_type": "vote", "action_trace": {}}},
            {"type": "semantic_repair_audit", "payload": {
                "repairable": True,
                "success": True,
                "target_preserved": True,
                "introduced_claim_count": 0,
            }},
            {"type": "action_trace_audit", "payload": {
                "phase": "hunter_shot",
                "action_trace": {
                    "final_action_type": "hunter_shot",
                    "power_role_evidence": {
                        "target_id": "p02",
                        "friendly_fire_risk": {"targets": []},
                        "retain_option": {"available": True},
                        "alternative_comparison": {
                            "legal_alternatives": [],
                            "no_legal_alternative": True,
                            "alternative_target": None,
                        },
                    },
                    "world_model_audit": {"possible_worlds": {"top_worlds": [
                        {"label": "w1", "key_assignments": {"p02": "werewolf"}, "why": ["event:1"]},
                        {"label": "w2", "key_assignments": {"p03": "werewolf"}, "why": ["claim:1"]},
                    ]}},
                },
            }},
            {"type": "reflection_complete", "payload": {"entries": [{
                "player_id": "p01",
                "decision_id": "reflection:g1:p01",
                "verification": {
                    "status": "verified",
                    "decision_id": "reflection:g1:p01",
                    "verified_fact_count": 0,
                    "verified_lessons": [],
                    "rejected_fact_count": 2,
                    "rejected_lesson_count": 1,
                },
            }]}},
        ],
        "deaths": [],
    }])

    assert report["terminal_post_win_game_model_call_count"] == 2
    assert report["semantic_repair_metrics_supported"] is True
    assert report["semantic_repair_success_rate"] == 1.0
    assert report["semantic_repair_target_preservation_rate"] == 1.0
    assert report["semantic_repair_no_new_claim_rate"] == 1.0
    assert report["possible_world_metrics_supported"] is True
    assert report["possible_world_unique_rate"] == 1.0
    assert report["possible_world_evidence_coverage_rate"] == 1.0
    assert report["power_role_evidence_metrics_supported"] is True
    assert report["power_role_evidence_completeness_rate"] == 1.0
    assert report["reflection_rejected_fact_count"] == 2
    assert report["reflection_rejected_lesson_count"] == 1

    empty = compute_balance_audit([])
    for supported_key, rate_keys in {
        "decision_execution_metrics_supported": (
            "reasoning_request_rate",
            "reasoning_confirmation_rate",
        ),
        "reasoning_fallback_keep_metrics_supported": (
            "reasoning_fallback_keep_rate",
        ),
        "semantic_repair_metrics_supported": (
            "semantic_repair_success_rate",
            "semantic_repair_target_preservation_rate",
            "semantic_repair_no_new_claim_rate",
        ),
        "possible_world_metrics_supported": (
            "possible_world_unique_rate",
            "possible_world_evidence_coverage_rate",
        ),
        "power_role_evidence_metrics_supported": (
            "power_role_evidence_completeness_rate",
        ),
    }.items():
        assert empty[supported_key] is False
        assert all(empty[key] is None for key in rate_keys)


def test_semantic_repair_rates_keep_invariants_as_independent_numerators() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    metrics = compute_acceptance_audit_metrics([{"events": [
        {"type": "semantic_repair_audit", "payload": {
            "repairable": True, "success": True,
            "target_preserved": False, "introduced_claim_count": 0,
        }},
        {"type": "semantic_repair_audit", "payload": {
            "repairable": True, "success": False,
            "target_preserved": True, "introduced_claim_count": 1,
        }},
    ]}])

    assert metrics["semantic_repair_success_count"] == 0
    assert metrics["semantic_repair_success_rate"] == 0.0
    assert metrics["semantic_repair_target_preservation_rate"] == 0.5
    assert metrics["semantic_repair_no_new_claim_rate"] == 0.5


def test_possible_world_uniqueness_is_grouped_per_prompt() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    shared = {"key_assignments": {"p02": "werewolf"}, "why": ["event:1"]}
    metrics = compute_acceptance_audit_metrics([{"game_id": "g1", "events": [
        {"type": "action_trace_audit", "payload": {
            "trace_id": "prompt-1", "action_trace": {"world_model_audit": {
                "possible_worlds": {"top_worlds": [shared, shared]}
            }},
        }},
        {"type": "action_trace_audit", "payload": {
            "trace_id": "prompt-2", "action_trace": {"world_model_audit": {
                "possible_worlds": {"top_worlds": [shared]}
            }},
        }},
    ]}])

    assert metrics["possible_world_prompt_count"] == 2
    assert metrics["possible_world_total_count"] == 3
    assert metrics["possible_world_unique_count"] == 2
    assert metrics["possible_world_unique_rate"] == 2 / 3


def test_power_role_evidence_requires_explicit_legal_alternative_target() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    def event(comparison, *, retain=True):
        evidence = {
            "target_id": "p02",
            "friendly_fire_risk": {"targets": []},
            "alternative_comparison": comparison,
        }
        if retain:
            evidence["retain_option"] = {"available": True}
        return {"type": "action_trace_audit", "payload": {"action_trace": {
            "final_action_type": "hunter_shot",
            "power_role_evidence": evidence,
        }}}

    metrics = compute_acceptance_audit_metrics([{"events": [
        event({
            "legal_alternatives": ["p03"],
            "no_legal_alternative": False,
            "alternative_target": "p03",
        }),
        event({
            "legal_alternatives": ["p03"],
            "no_legal_alternative": False,
            "alternative_target": None,
        }),
        event({
            "legal_alternatives": [],
            "no_legal_alternative": True,
            "alternative_target": None,
        }),
        event({
            "legal_alternatives": [],
            "no_legal_alternative": True,
        }),
        event({
            "legal_alternatives": [],
            "no_legal_alternative": True,
            "alternative_target": None,
        }, retain=False),
    ]}])

    assert metrics["power_role_evidence_complete_count"] == 2
    assert metrics["power_role_evidence_completeness_rate"] == 0.4


def test_wolf_normalization_rate_uses_triggered_normalization_denominator() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_wolf_plan_outcome_metrics

    metrics = compute_wolf_plan_outcome_metrics([{"game_id": "g1", "events": [
        {"type": "wolf_team_plan", "payload": {
            "decision_id": "normalized",
            "normalization_repairs": ["truncate:reasoning"],
        }},
        {"type": "wolf_team_plan", "payload": {"decision_id": "direct"}},
        {"type": "wolf_team_plan_fallback", "payload": {
            "decision_id": "failed", "reason": "schema_validation_failed",
        }},
    ]}])

    assert metrics["wolf_team_plan_normalization_triggered_count"] == 1
    assert metrics["wolf_team_plan_normalization_metrics_supported"] is True
    assert metrics["wolf_team_plan_normalization_success_rate"] == 1.0
    assert metrics["wolf_plan_schema_fallback_rate"] == 1 / 3
    assert metrics["wolf_plan_strategy_fallback_rate"] == 0.0


def test_wolf_normalization_failed_after_repair_stays_in_triggered_denominator() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_wolf_plan_outcome_metrics

    metrics = compute_wolf_plan_outcome_metrics([{"game_id": "g1", "events": [
        {"type": "wolf_team_plan_fallback", "payload": {
            "decision_id": "failed-normalized",
            "reason": "schema_validation_failed",
            "normalization_triggered": True,
            "normalization_repairs": ["unwrap:night_plan"],
        }},
        {"type": "wolf_team_plan", "payload": {
            "decision_id": "failed-normalized",
            "consensus_method": "fallback",
            "normalization_triggered": True,
            "normalization_repairs": ["unwrap:night_plan"],
        }},
    ]}])

    assert metrics["wolf_team_plan_normalization_triggered_count"] == 1
    assert metrics["wolf_team_plan_normalization_success_count"] == 0
    assert metrics["wolf_team_plan_normalization_success_rate"] == 0.0
