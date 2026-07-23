# -*- coding: utf-8 -*-
"""
验证验收报告只消费强类型执行事实，并为所有比率保留精确分母。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-19
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

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


def _world_proof(assignments: dict[str, str]) -> dict[str, object]:
    pairs = [[player, role] for player, role in sorted(assignments.items())]
    canonical = json.dumps(
        pairs, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return {
        "world_id": f"world_{hashlib.sha256(canonical).hexdigest()}",
        "canonical_assignment": pairs,
    }


def _semantic_pair(
    payload: dict[str, object],
    *,
    action_index: int,
    game_id: str = "g1",
) -> list[dict[str, object]]:
    payload = {
        "speaker_attribution_preserved": True,
        "negation_preserved": True,
        **payload,
    }
    identity = {
        "trace_id": f"trace-{action_index}", "game_id": game_id,
        "action_index": action_index, "task_type": "speech",
    }
    return [
        {"type": "semantic_repair_audit", "payload": {**payload, **identity}},
        {"type": "action_trace_audit", "payload": {
            **identity, "action_trace": {"semantic_repair_audit": payload},
        }},
    ]


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
        "attempt_count": len(attempts),
        "retry_count": sum(
            item.route_kind is RouteKind.RETRY for item in attempts
        ),
        "provider_fallback_count": sum(
            item.route_kind is RouteKind.PROVIDER_FALLBACK for item in attempts
        ),
        "generated_by": (
            "terminal_fallback"
            if decision == "terminal_fallback"
            else "repair"
            if decision == "repaired_success"
            else "provider_fallback"
            if decision == "provider_fallback_success"
            else "model"
        ),
        "terminal_failure_code": (
            attempts[-1].root_cause.value
            if decision == "terminal_fallback" else None
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
    assert metrics["retry_count"] == sum(
        item.route_kind is RouteKind.RETRY for item in attempts
    )
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
    assert metrics["runtime_timeout_count"] == 1


def test_execution_report_deduplicates_wolf_consensus_trace_projection() -> None:
    """同一狼队决策的独立审计和共识副本只能投影一次。"""
    from werewolf_agent.evaluation.balance_audit import (
        compute_decision_execution_metrics,
    )

    attempts = (
        _attempt(
            1, RouteKind.PRIMARY, AttemptOutcome.FAILURE,
            cause=RootCause.TIMEOUT,
        ),
        _attempt(2, RouteKind.RETRY, AttemptOutcome.SUCCESS),
    )
    individual_trace = {
        "execution_attempts": attempts,
        "attempt_count": 2,
        "retry_count": 1,
        "provider_fallback_count": 0,
        "runtime_timeout_count": 1,
        "generated_by": "model",
        "terminal_failure_code": None,
    }
    # runtime/nodes/wolf_consensus.py 会把同一 trace 同时写入独立审计
    # 事件和 wolf_kill_selected.action_traces。
    nested_trace = dict(individual_trace)

    metrics = compute_decision_execution_metrics([{
        "game_id": "g1",
        "events": [
            {"type": "action_trace_audit", "payload": {
                "player_id": "p02",
                "phase": "wolf_consensus",
                "task_type": "wolf_consensus",
                "day_number": 0,
                "night_number": 1,
                "trace_id": "g1:p02:wolf_consensus:D0:N1:wolf_consensus:0",
                "game_id": "g1",
                "action_index": 0,
                "action_trace": individual_trace,
            }},
            {"type": "wolf_kill_selected", "payload": {
                "night_number": 1,
                "target_id": "p01",
                "action_traces": {"p02": nested_trace},
            }},
        ],
    }])

    assert metrics["decision_count"] == 1
    assert metrics["attempt_count"] == 2
    assert metrics["retry_count"] == 1
    assert metrics["provider_fallback_count"] == 0
    assert metrics["runtime_timeout_count"] == 1
    assert metrics["root_cause_counts"] == {"none": 1, "timeout": 1}
    assert metrics["decision_outcome_counts"] == {"retry_success": 1}
    assert metrics["attempt_retry_consistency_error_count"] == 0


def test_execution_report_merges_richer_wolf_trace_without_losing_audit_context() -> None:
    """同组副本要保留完整 attempts、timeout 校验和独立审计任务上下文。"""
    from werewolf_agent.evaluation.balance_audit import (
        compute_decision_execution_metrics,
    )

    timeout = _attempt(
        1, RouteKind.PRIMARY, AttemptOutcome.FAILURE,
        cause=RootCause.TIMEOUT,
    )
    repair_success = _attempt(2, RouteKind.REPAIR, AttemptOutcome.SUCCESS)
    retry_failure = _attempt(
        2, RouteKind.RETRY, AttemptOutcome.FAILURE,
        cause=RootCause.PROVIDER_ERROR,
    )
    final_repair_success = _attempt(
        3, RouteKind.REPAIR, AttemptOutcome.SUCCESS,
    )
    individual_trace = {
        "execution_attempts": (timeout, repair_success),
        "attempt_count": 2,
        "retry_count": 0,
        "provider_fallback_count": 0,
        "runtime_timeout_count": 1,
        "generated_by": "model",
        "terminal_failure_code": None,
    }
    nested_trace = {
        "execution_attempts": (timeout, retry_failure, final_repair_success),
        "attempt_count": 3,
        "retry_count": 1,
        "provider_fallback_count": 0,
        "generated_by": "model",
        "terminal_failure_code": None,
    }
    metrics = compute_decision_execution_metrics([{
        "game_id": "g1",
        "events": [
            {"type": "action_trace_audit", "payload": {
                "player_id": "p02",
                "phase": "wolf_consensus",
                "task_type": "wolf_team_plan",
                "trace_id": "g1:p02:wolf_consensus:D0:N1:wolf_team_plan:0",
                "action_trace": individual_trace,
            }},
            {"type": "wolf_kill_selected", "payload": {
                "night_number": 1,
                "target_id": "p01",
                "action_traces": {"p02": nested_trace},
            }},
        ],
    }])

    assert metrics["decision_count"] == 1
    assert metrics["attempt_count"] == 3
    assert metrics["retry_count"] == 1
    assert metrics["runtime_timeout_count"] == 1
    assert metrics["attempt_retry_consistency_error_count"] == 0
    assert metrics["reasoning_task_type_missing_count"] == 0
    assert metrics["critical_task_reasoning_request_count"] == 3
    assert metrics["critical_task_reasoning_requested_count"] == 3
    assert metrics["critical_task_reasoning_status_request_count"] == 1
    assert metrics["critical_task_reasoning_status_explicit_count"] == 1


def test_execution_report_keeps_equal_traces_with_distinct_request_ids() -> None:
    """相同业务内容但不同请求身份仍是两次决策。"""
    from werewolf_agent.evaluation.balance_audit import (
        compute_decision_execution_metrics,
    )

    first_attempts = (
        _attempt(1, RouteKind.PRIMARY, AttemptOutcome.SUCCESS),
    )
    second_attempts = (
        replace(
            first_attempts[0],
            opaque_request_id=OpaqueRequestId.new("game", "11223344"),
        ),
    )
    trace_template = {
        "attempt_count": 1,
        "retry_count": 0,
        "provider_fallback_count": 0,
        "runtime_timeout_count": 0,
        "generated_by": "model",
        "terminal_failure_code": None,
    }
    metrics = compute_decision_execution_metrics([{
        "game_id": "g1",
        "events": [
            {"type": "action_trace_audit", "payload": {
                "trace_id": "g1:p02:wolf_consensus:D0:N1:wolf_consensus:0",
                "action_trace": {
                    **trace_template,
                    "execution_attempts": first_attempts,
                },
            }},
            {"type": "action_trace_audit", "payload": {
                "trace_id": "g1:p02:wolf_consensus:D0:N2:wolf_consensus:1",
                "action_trace": {
                    **trace_template,
                    "execution_attempts": second_attempts,
                },
            }},
        ],
    }])

    assert metrics["decision_count"] == 2
    assert metrics["attempt_count"] == 2
    assert metrics["runtime_timeout_count"] == 0
    assert metrics["decision_outcome_counts"] == {"direct_success": 2}


def test_execution_report_fails_closed_on_conflicting_audit_task_context() -> None:
    """同一请求的独立审计任务冲突时不能任选一个关键任务。"""
    from werewolf_agent.evaluation.balance_audit import (
        compute_decision_execution_metrics,
    )

    trace = {
        "execution_attempts": (
            _attempt(1, RouteKind.PRIMARY, AttemptOutcome.SUCCESS),
        ),
        "attempt_count": 1,
        "retry_count": 0,
        "provider_fallback_count": 0,
        "runtime_timeout_count": 0,
        "generated_by": "model",
        "terminal_failure_code": None,
    }
    metrics = compute_decision_execution_metrics([{
        "events": [
            {"type": "action_trace_audit", "payload": {
                "player_id": "p02",
                "task_type": "wolf_team_plan",
                "trace_id": "trace-1",
                "action_trace": trace,
            }},
            {"type": "action_trace_audit", "payload": {
                "player_id": "p02",
                "task_type": "speech",
                "trace_id": "trace-2",
                "action_trace": dict(trace),
            }},
        ],
    }])

    assert metrics["decision_count"] == 1
    assert metrics["reasoning_task_type_missing_count"] == 1
    assert metrics["critical_task_reasoning_request_count"] == 0


def test_execution_report_aggregates_only_real_provider_timeouts() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    timeout = _attempt(
        1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT,
    )
    success = _attempt(2, RouteKind.RETRY, AttemptOutcome.SUCCESS)
    traces = [
        {"execution_attempts": (timeout, success)},
        {"execution_attempts": (
            _attempt(
                1, RouteKind.PRIMARY, AttemptOutcome.FAILURE,
                cause=RootCause.PROVIDER_ERROR,
            ),
            _attempt(
                2, RouteKind.PROVIDER_FALLBACK, AttemptOutcome.SUCCESS,
                provider="backup",
            ),
        )},
        {"execution_attempts": (
            _attempt(
                1, RouteKind.PRIMARY, AttemptOutcome.FAILURE,
                cause=RootCause.INVALID_OUTPUT,
            ),
            _attempt(2, RouteKind.REPAIR, AttemptOutcome.SUCCESS),
        )},
        {"execution_attempts": (
            _attempt(
                1, RouteKind.PRIMARY, AttemptOutcome.FAILURE,
                cause=RootCause.PROVIDER_ERROR,
            ),
            _attempt(
                2, RouteKind.SAFE_FALLBACK, AttemptOutcome.FAILURE,
                cause=RootCause.TIMEOUT,
            ),
        )},
        {"execution_attempts": (
            replace(timeout, provider_attempted=False), success,
        )},
    ]

    metrics = compute_decision_execution_metrics([
        {"events": [{"type": "action_trace_audit", "payload": {
            "action_trace": trace,
        }} for trace in traces[:2]]},
        {"events": [{"type": "action_trace_audit", "payload": {
            "action_trace": trace,
        }} for trace in traces[2:]]},
    ])

    assert metrics["runtime_timeout_count"] == 1


def test_execution_report_backfills_legacy_timeout_count_and_records_explicit_drift() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    timeout = _attempt(
        1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT,
    )
    success = _attempt(2, RouteKind.RETRY, AttemptOutcome.SUCCESS)
    second_timeout = replace(
        timeout,
        opaque_request_id=OpaqueRequestId.new("game", "55667788"),
    )
    second_success = replace(
        success,
        opaque_request_id=OpaqueRequestId.new("game", "55667788"),
    )
    metrics = compute_decision_execution_metrics([{
        "events": [
            {"type": "action_trace_audit", "payload": {"action_trace": {
                "execution_attempts": (timeout, success),
            }}},
            {"type": "action_trace_audit", "payload": {"action_trace": {
                "runtime_timeout_count": 0,
            }}},
            {"type": "action_trace_audit", "payload": {"action_trace": {}}},
            {"type": "action_trace_audit", "payload": {"action_trace": {
                "runtime_timeout_count": 1,
            }}},
            {"type": "action_trace_audit", "payload": {"action_trace": {
                "execution_attempts": (second_timeout, second_success),
                "runtime_timeout_count": 0,
            }}},
        ],
    }])

    assert metrics["runtime_timeout_count"] == 2
    assert metrics["attempt_retry_consistency_error_count"] == 2


@pytest.mark.parametrize("provider_attempted", [0, 1, None, "true"])
def test_execution_report_rejects_malformed_provider_attempted_timeout(
    provider_attempted: object,
) -> None:
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    metrics = compute_decision_execution_metrics([{
        "events": [{"type": "action_trace_audit", "payload": {
            "action_trace": {"execution_attempts": [
                {
                    "opaque_request_id": {"value": "game:abcdef12"},
                    "ordinal": 1,
                    "provider": "primary",
                    "model": "model-a",
                    "route_kind": "primary",
                    "root_cause": "timeout",
                    "attempt_outcome": "attempt_failure",
                    "requested_reasoning_level": "high",
                    "normalized_reasoning_status": "confirmed",
                    "reasoning_token_count": 1,
                    "evidence_kind": "token_count",
                    "provider_attempted": provider_attempted,
                },
                {
                    "opaque_request_id": {"value": "game:abcdef12"},
                    "ordinal": 2,
                    "provider": "primary",
                    "model": "model-a",
                    "route_kind": "retry",
                    "root_cause": "none",
                    "attempt_outcome": "attempt_success",
                    "requested_reasoning_level": "high",
                    "normalized_reasoning_status": "confirmed",
                    "reasoning_token_count": 1,
                    "evidence_kind": "token_count",
                },
            ]},
        }}],
    }])

    assert metrics["runtime_timeout_count"] == 0
    assert metrics["decision_execution_invalid_sequence_count"] == 1


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
                "attempt_count": 2,
                "provider_fallback_count": 0,
                "generated_by": "model",
                "terminal_failure_code": None,
            },
        }}],
    }])

    assert metrics["attempt_retry_consistency_error_count"] == 0


def test_provider_fallback_is_not_counted_as_retry() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    attempts = (
        _attempt(
            1,
            RouteKind.PRIMARY,
            AttemptOutcome.FAILURE,
            cause=RootCause.PROVIDER_ERROR,
        ),
        _attempt(
            2,
            RouteKind.PROVIDER_FALLBACK,
            AttemptOutcome.SUCCESS,
            provider="backup",
        ),
    )
    metrics = compute_decision_execution_metrics([{
        "events": [{"type": "action_trace_audit", "payload": {
            "task_type": "vote",
            "action_trace": {
                "execution_attempts": attempts,
                "attempt_count": 2,
                "retry_count": 0,
                "provider_fallback_count": 1,
                "generated_by": "provider_fallback",
                "terminal_failure_code": None,
            },
        }}],
    }])

    assert metrics["attempt_count"] == 2
    assert metrics["retry_count"] == 0
    assert metrics["provider_fallback_count"] == 1
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
        reasoning_token_count=0,
        evidence_kind=EvidenceKind.NONE,
    )
    speech_attempt = replace(
        medium_attempt,
        opaque_request_id=OpaqueRequestId.new("game", "55667788"),
    )
    judge_attempt = replace(
        none_attempt,
        opaque_request_id=OpaqueRequestId.new("game", "99aabbcc"),
    )
    untyped_attempt = replace(
        medium_attempt,
        opaque_request_id=OpaqueRequestId.new("game", "ddeeff00"),
    )
    games = [{"events": [
        {"type": "action_trace_audit", "payload": {
            "task_type": "wolf_team_plan",
            "action_trace": {"execution_attempts": (medium_attempt,)},
        }},
            {"type": "action_trace_audit", "payload": {
                "task_type": "speech",
                "action_trace": {"execution_attempts": (speech_attempt,)},
            }},
            {"type": "action_trace_audit", "payload": {
                "task_type": "judge_phase",
                "action_trace": {"execution_attempts": (judge_attempt,)},
            }},
            {"type": "action_trace_audit", "payload": {
                "action_trace": {"execution_attempts": (untyped_attempt,)},
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


def test_critical_reasoning_checks_every_attempt_and_counts_unknown_task() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    high = _attempt(
        1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT
    )
    low = replace(
        _attempt(2, RouteKind.RETRY, AttemptOutcome.SUCCESS),
        requested_reasoning_level=ReasoningLevel.NONE,
        normalized_reasoning_status=ReasoningStatus.NOT_REQUESTED,
        reasoning_token_count=0,
        evidence_kind=EvidenceKind.NONE,
    )
    metrics = compute_decision_execution_metrics([{"events": [
        {"type": "action_trace_audit", "payload": {
            "task_type": "vote",
            "action_trace": {"execution_attempts": (high, low)},
        }},
        {"type": "action_trace_audit", "payload": {
            "task_type": "future_unknown_task",
            "action_trace": {"execution_attempts": (replace(high, ordinal=1),)},
        }},
    ]}])

    assert metrics["critical_task_reasoning_request_count"] == 2
    assert metrics["critical_task_reasoning_requested_count"] == 1
    assert metrics["critical_task_reasoning_request_coverage"] == 0.5
    assert metrics["reasoning_task_type_unknown_count"] == 1


def test_critical_reasoning_effective_request_rejects_terminal_statuses() -> None:
    """关键任务只有达到 policy 且未被 provider 拒绝才算有效请求。"""
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    statuses = (
        ReasoningStatus.REQUESTED_UNCONFIRMED,
        ReasoningStatus.UNSUPPORTED,
        ReasoningStatus.FALLBACK_DISABLED,
    )
    evidence = (
        EvidenceKind.FIELD_ECHO,
        EvidenceKind.UNSUPPORTED,
        EvidenceKind.FALLBACK_DISABLED,
    )
    attempts = tuple(
        replace(
            _attempt(index, RouteKind.PRIMARY if index == 1 else RouteKind.RETRY,
                     AttemptOutcome.FAILURE if index < 3 else AttemptOutcome.SUCCESS,
                     cause=RootCause.TIMEOUT if index < 3 else RootCause.NONE,
                     reasoning_status=status),
            normalized_reasoning_status=status,
            reasoning_token_count=0,
            evidence_kind=evidence_kind,
        )
        for index, (status, evidence_kind) in enumerate(
            zip(statuses, evidence, strict=True), start=1
        )
    )
    metrics = compute_decision_execution_metrics([{"events": [{
        "type": "action_trace_audit",
        "payload": {
            "task_type": "wolf_team_plan",
            "action_trace": {"execution_attempts": attempts},
        },
    }]}])

    assert metrics["critical_task_reasoning_effective_request_count"] == 1
    assert metrics["critical_task_reasoning_effective_request_rate"] == 1 / 3
    assert metrics["critical_task_reasoning_unsupported_count"] == 1
    assert metrics["critical_task_reasoning_fallback_disabled_count"] == 1
    assert metrics["reasoning_unconfirmed_count"] == 1
    assert metrics["reasoning_confirmation_rate"] == 0.0


def test_unconfirmed_reasoning_is_dispatched_but_requires_confirmation_review() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    unconfirmed = replace(
        _attempt(1, RouteKind.PRIMARY, AttemptOutcome.SUCCESS),
        normalized_reasoning_status=ReasoningStatus.REQUESTED_UNCONFIRMED,
        reasoning_token_count=0,
        evidence_kind=EvidenceKind.FIELD_ECHO,
    )
    metrics = compute_decision_execution_metrics([{"events": [{
        "type": "action_trace_audit", "payload": {
            "task_type": "vote",
            "action_trace": {"execution_attempts": (unconfirmed,)},
        },
    }]}])

    assert metrics["critical_task_reasoning_request_dispatched_rate"] == 1.0
    assert metrics["critical_task_reasoning_confirmed_rate"] == 0.0
    assert metrics["critical_task_reasoning_requires_confirmation_review_count"] == 1
    assert metrics["critical_task_reasoning_effective_request_rate"] == 1.0


def test_provider_fallback_keep_requires_critical_task_policy_minimum() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_decision_execution_metrics

    first = _attempt(
        1,
        RouteKind.PRIMARY,
        AttemptOutcome.FAILURE,
        cause=RootCause.PROVIDER_ERROR,
    )
    medium_fallback = replace(
        _attempt(
            2,
            RouteKind.PROVIDER_FALLBACK,
            AttemptOutcome.SUCCESS,
            provider="backup",
            reasoning_status=ReasoningStatus.REQUESTED_UNCONFIRMED,
        ),
        requested_reasoning_level=ReasoningLevel.MEDIUM,
    )
    judge_first = replace(
        first,
        opaque_request_id=OpaqueRequestId.new("game", "1234abcd"),
    )
    judge_fallback = replace(
        medium_fallback,
        opaque_request_id=OpaqueRequestId.new("game", "1234abcd"),
    )
    metrics = compute_decision_execution_metrics([{"events": [
        {
            "type": "action_trace_audit",
            "payload": {
                "task_type": "wolf_team_plan",
                "action_trace": {"execution_attempts": (first, medium_fallback)},
            },
        },
        {
                "type": "action_trace_audit",
                "payload": {
                    "task_type": "judge_phase",
                    "action_trace": {
                        "execution_attempts": (judge_first, judge_fallback),
                    },
                },
        },
    ]}])

    assert metrics["reasoning_fallback_keep_metrics_supported"] is True
    assert metrics["reasoning_fallback_request_count"] == 1
    assert metrics["reasoning_fallback_keep_count"] == 0
    assert metrics["reasoning_fallback_keep_rate"] == 0.0


def test_acceptance_metrics_use_exact_denominators_and_explicit_support() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    world_one = _world_proof({"p01": "hunter", "p02": "werewolf", "p03": "villager"})
    world_two = _world_proof({"p01": "hunter", "p02": "villager", "p03": "werewolf"})
    report = compute_balance_audit([{
        "game_id": "g1",
        "winning_faction": "good",
        "players": {
            "p01": {"role": "hunter"}, "p02": {"role": "werewolf"},
            "p03": {"role": "villager"},
        },
        "events": [
            {"type": "victory", "payload": {"winner": "good"}},
            {"type": "action_trace_audit", "payload": {"task_type": "vote", "action_trace": {}}},
            {"type": "semantic_repair_audit", "payload": {
                "repairable": True,
                "success": True,
                "target_preserved": True,
                "speaker_attribution_preserved": True,
                "negation_preserved": True,
                "introduced_claim_count": 0,
                "verified_claim_count": 0,
                "retained_verified_claim_count": 0,
                "generic_template_used": False,
                "fallback_kind": "no_fallback",
                "trace_id": "trace-semantic", "game_id": "g1",
                "action_index": 7, "task_type": "speech",
            }},
            {"type": "action_trace_audit", "payload": {
                "trace_id": "trace-semantic", "game_id": "g1",
                "action_index": 7, "task_type": "speech",
                "action_trace": {"semantic_repair_audit": {
                    "repairable": True, "success": True,
                        "target_preserved": True,
                        "speaker_attribution_preserved": True,
                        "negation_preserved": True,
                        "introduced_claim_count": 0,
                    "verified_claim_count": 0, "retained_verified_claim_count": 0,
                    "generic_template_used": False, "fallback_kind": "no_fallback",
                }},
            }},
            {"type": "speech", "payload": {
                "speaker": "p02", "text": "我是狼人。",
                "claims": [{"type": "role", "value": "werewolf"}],
            }},
            {"type": "speech", "payload": {
                "speaker": "p03", "text": "我是狼人。",
                "claims": [{"type": "role", "value": "werewolf"}],
            }},
            {"type": "action_trace_audit", "payload": {
                "player_id": "p01",
                "phase": "hunter_shot",
                "action_trace": {
                    "final_action_type": "hunter_shot",
                        "power_role_evidence": {
                            "target_id": "p02",
                            "target_evidence": {
                                "selected_score": 5,
                                "selected_signals": ["seer_check_wolf"],
                            },
                            "target_comparison": {
                                "selected_score": 5,
                                "selected_signals": ["seer_check_wolf"],
                                "alternative_target": None,
                                "alternative_score": None,
                                "alternative_signals": [],
                                "comparison_basis": "ranked public evidence score",
                            },
                        "friendly_fire_risk": {
                            "status": "assessed", "targets": [], "basis": "无误伤目标",
                        },
                        "retain_option": {
                            "action": "no_action", "available": True,
                            "required": False, "reason": "可保留不开枪",
                        },
                        "alternative_comparison": {
                            "legal_alternatives": [],
                            "no_legal_alternative": True,
                            "alternative_target": None,
                        },
                    },
                    "world_model_audit": {"possible_worlds": {"top_worlds": [
                        {"label": world_one["world_id"], "key_assignments": {"p02": "werewolf"}, "why": ["claim:g1:4"]},
                        {"label": world_two["world_id"], "key_assignments": {"p03": "werewolf"}, "why": ["claim:g1:5"]},
                    ]}, "authoritative_world_identities": [world_one, world_two],
                    "public_evidence_ids": ["claim:g1:4", "claim:g1:5"]},
                },
            }},
            {"type": "player_died", "payload": {
                "player_id": "p02", "reason": "hunter_shot", "source_player_id": "p01",
            }},
            {"type": "reflection_complete", "payload": {"player_count": 1, "entries": [{
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
        "deaths": [{
            "player_id": "p02", "reason": "hunter_shot", "source_player_id": "p01",
        }],
    }])

    assert report["terminal_post_win_game_model_call_count"] == 3
    assert report["semantic_repair_metrics_supported"] is True
    assert report["semantic_repair_success_rate"] == 1.0
    assert report["semantic_repair_target_preservation_rate"] == 1.0
    assert report["semantic_repair_speaker_attribution_preservation_rate"] == 1.0
    assert report["semantic_repair_negation_preservation_rate"] == 1.0
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
            "semantic_repair_speaker_attribution_preservation_rate",
            "semantic_repair_negation_preservation_rate",
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

    first = {
            "repairable": True, "success": True,
            "target_preserved": False, "introduced_claim_count": 0,
                "verified_claim_count": 1, "retained_verified_claim_count": 0,
                "generic_template_used": False,
                "fallback_kind": "no_fallback",
    }
    second = {
            "repairable": True, "success": False,
            "target_preserved": True, "introduced_claim_count": 1,
                "verified_claim_count": 1, "retained_verified_claim_count": 1,
                "generic_template_used": True,
                "fallback_kind": "generic_template",
    }
    metrics = compute_acceptance_audit_metrics([{"game_id": "g1", "players": {
        "p01": {"role": "villager"}, "p02": {"role": "werewolf"},
    }, "events": [
        *_semantic_pair(first, action_index=1),
        *_semantic_pair(second, action_index=2),
    ]}])

    assert metrics["semantic_repair_success_count"] == 0
    assert metrics["semantic_repair_success_rate"] == 0.0
    assert metrics["semantic_repair_target_preservation_rate"] == 0.5
    assert metrics["semantic_repair_no_new_claim_rate"] == 0.5
    assert metrics["semantic_repair_verified_claim_retention_metrics_supported"] is True
    assert metrics["semantic_repair_verified_claim_retention_rate"] == 0.5
    assert metrics["semantic_repair_generic_template_count"] == 1


def test_balance_audit_propagates_v2_public_evidence_safety_metrics() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    report = compute_balance_audit([{
        "game_id": "v2-semantic-report",
        "winning_faction": "good",
        "players": {"p01": {"role": "villager"}},
        "events": _semantic_pair(
            {
                "repairable": True,
                "semantic_gate_version": 2,
                "success": True,
                "target_preserved": True,
                "introduced_claim_count": 0,
                "unsupported_public_claim_count": 0,
                "verified_claim_count": 0,
                "retained_verified_claim_count": 0,
                "generic_template_used": False,
                "fallback_kind": "no_fallback",
            },
            action_index=1,
            game_id="v2-semantic-report",
        ),
    }])

    assert report["semantic_repair_public_evidence_safety_metrics_supported"] is True
    assert report["semantic_repair_public_evidence_safety_rate"] == 1.0


def test_report_exports_threshold_support_metrics() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_balance_audit

    report = compute_balance_audit([{
        "winning_faction": "good",
        "players": {},
        "events": [
            {"type": "reflection_complete", "payload": {
                "player_count": 0, "entries": [],
            }},
                {"type": "reflection_persistence_audit", "payload": {
                    "expected_entry_count": 0,
                    "persistence_complete": True,
                    "rollback_complete": True,
                    "entries": [],
            }},
        ],
    }])

    assert report["completion_rate"] == 1.0
    assert report["persona_exposure_linkage_rate"] is None
    assert report["critical_task_reasoning_status_metrics_supported"] is False
    assert report["critical_task_reasoning_status_explicit_rate"] is None
    assert report["reflection_contamination_metrics_supported"] is False
    assert report["reflection_persisted_rejected_fact_count"] is None
    assert report["acceptance_projection_unsupported_reason"] == "missing_game_id"


def test_semantic_and_reflection_metrics_fail_closed_when_source_fields_missing() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    metrics = compute_acceptance_audit_metrics([{"events": [
        {"type": "semantic_repair_audit", "payload": {
            "repairable": True,
            "success": True,
            "target_preserved": True,
        }},
        {"type": "reflection_complete", "payload": {"entries": [{
            "player_id": "p01",
            "verification": {"rejected_fact_count": 2},
        }]}},
    ]}])

    assert metrics["semantic_repair_metrics_supported"] is False
    assert metrics["semantic_repair_no_new_claim_rate"] is None
    assert metrics["semantic_repair_generic_template_count"] is None
    assert metrics["semantic_repair_verified_claim_retention_metrics_supported"] is False
    assert metrics["semantic_repair_verified_claim_retention_rate"] is None
    assert metrics["reflection_contamination_metrics_supported"] is False
    assert metrics["reflection_persisted_rejected_fact_count"] is None


def test_reflection_metrics_fail_closed_when_expected_persistence_is_incomplete() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    metrics = compute_acceptance_audit_metrics([{"events": [{
        "type": "reflection_persistence_audit",
        "payload": {
            "persistence_complete": False,
            "entries": [{
                "player_id": "p01",
                "decision_id": "reflection:g1:p01",
                "entry_id": "reflection_g1_p01",
                "row_found": False,
                "persistence_complete": False,
                "persisted_rejected_fact_count": 0,
            }],
        },
    }]}])

    assert metrics["reflection_contamination_metrics_supported"] is False
    assert metrics["reflection_persisted_rejected_fact_count"] is None


def test_reflection_metrics_fail_closed_when_one_game_omits_expected_audit() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    missing_audit_game = {"game_id": "g1", "events": [{
        "type": "reflection_complete",
        "payload": {"entries": [{
            "player_id": "p01",
            "verification": {
                "decision_id": "reflection:g1:p01",
                "verified_lessons": [{
                    "lesson_id": "l1",
                    "abstraction": "对跳时先核验公开票型",
                }],
                "rejected_fact_count": 0,
                "rejected_lesson_count": 0,
            },
        }]},
    }]}
    clean_game = {"game_id": "g2", "events": [{
        "type": "reflection_persistence_audit",
        "payload": {
            "persistence_complete": True,
            "entries": [{
                "player_id": "p02",
                "decision_id": "reflection:g2:p02",
                "entry_id": "reflection_g2_p02",
                "row_found": True,
                "persistence_complete": True,
                "persisted_rejected_fact_count": 0,
            }],
        },
    }]}

    metrics = compute_acceptance_audit_metrics([missing_audit_game, clean_game])

    assert metrics["reflection_contamination_metrics_supported"] is False
    assert metrics["reflection_persisted_rejected_fact_count"] is None


def test_possible_world_uniqueness_is_grouped_per_prompt() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    proof = _world_proof({"p01": "villager", "p02": "werewolf"})
    shared = {
        "label": proof["world_id"],
        "key_assignments": {"p02": "werewolf"},
        "why": ["claim:g1:0"],
    }
    metrics = compute_acceptance_audit_metrics([{"game_id": "g1", "players": {
        "p01": {"role": "villager"}, "p02": {"role": "werewolf"},
    }, "events": [
        {"type": "speech", "payload": {
            "speaker": "p02", "text": "我是狼人。",
            "claims": [{"type": "role", "value": "werewolf"}],
        }},
        {"type": "action_trace_audit", "payload": {
            "trace_id": "prompt-1", "action_trace": {"world_model_audit": {
                "possible_worlds": {"top_worlds": [shared, shared]},
                "authoritative_world_identities": [proof],
                "public_evidence_ids": ["claim:g1:0"],
            }},
        }},
        {"type": "action_trace_audit", "payload": {
            "trace_id": "prompt-2", "action_trace": {"world_model_audit": {
                "possible_worlds": {"top_worlds": [shared]},
                "authoritative_world_identities": [proof],
                "public_evidence_ids": ["claim:g1:0"],
            }},
        }},
    ]}])

    assert metrics["possible_world_prompt_count"] == 2
    assert metrics["possible_world_total_count"] == 3
    assert metrics["possible_world_unique_count"] == 1
    assert metrics["possible_world_unique_rate"] == 1 / 3


def test_possible_world_uniqueness_rejects_missing_ids_and_duplicate_assignments() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    proof = _world_proof({"p01": "villager", "p02": "werewolf"})
    worlds = [
        {"key_assignments": {"p02": "werewolf"}, "why": ["claim:g1:0"]},
        {"label": proof["world_id"], "key_assignments": {"p02": "werewolf"}, "why": ["claim:g1:0"]},
        {"world_id": "world-c", "key_assignments": {"p03": "seer"}, "why": ["claim:g1:1"]},
    ]
    metrics = compute_acceptance_audit_metrics([{"game_id": "g1", "players": {
        "p01": {"role": "villager"}, "p02": {"role": "werewolf"},
    }, "events": [
        {"type": "speech", "payload": {
            "speaker": "p02", "text": "我是狼人。",
            "claims": [{"type": "role", "value": "werewolf"}],
        }},
        {"type": "speech", "payload": {
            "speaker": "p03", "text": "我是预言家。",
            "claims": [{"type": "role", "value": "seer"}],
        }},
        {"type": "action_trace_audit", "payload": {"action_trace": {
            "world_model_audit": {
                "possible_worlds": {"top_worlds": worlds},
                "authoritative_world_identities": [proof],
                "public_evidence_ids": ["claim:g1:0", "claim:g1:1"],
            },
        }}},
    ]}])

    assert metrics["possible_world_total_count"] == 3
    assert metrics["possible_world_unique_count"] == 1
    assert metrics["possible_world_unique_rate"] == 1 / 3


def test_possible_world_evidence_rejects_unknown_cross_game_and_private_refs() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    worlds = [
        {"label": "valid", "key_assignments": {"p02": "werewolf"}, "why": ["claim:g1:0"]},
        {"label": "unknown", "key_assignments": {"p03": "werewolf"}, "why": ["claim:g1:99"]},
        {"label": "cross", "key_assignments": {"p04": "werewolf"}, "why": ["claim:g2:0"]},
        {"label": "private", "key_assignments": {"p05": "werewolf"}, "why": ["claim:g1:1"]},
    ]
    metrics = compute_acceptance_audit_metrics([{
        "game_id": "g1",
        "players": {f"p0{i}": {"role": "villager"} for i in range(1, 7)},
        "events": [
        {"type": "speech", "payload": {
            "speaker": "p02", "text": "我是狼人。",
            "claims": [{"type": "role", "value": "werewolf"}],
        }},
        {"type": "speech", "payload": {
            "speaker": "p06", "text": "私密", "visibility": "moderator_only",
        }},
        {"type": "action_trace_audit", "payload": {"action_trace": {
            "world_model_audit": {
                "possible_worlds": {"top_worlds": worlds},
                "public_evidence_ids": [
                    "claim:g1:0", "claim:g1:99", "claim:g2:0", "claim:g1:1",
                ],
            },
        }}},
    ]}])

    assert metrics["possible_world_evidence_covered_count"] == 1
    assert metrics["possible_world_evidence_coverage_rate"] == 0.25


def test_possible_world_evidence_rejects_public_suspicion_as_assignment_support() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    metrics = compute_acceptance_audit_metrics([{
        "game_id": "g1",
        "players": {"p01": {"role": "villager"}, "p02": {"role": "werewolf"}},
        "events": [
        {"type": "speech", "payload": {
            "speaker": "p01", "text": "我怀疑p02，他的发言很可疑。",
        }},
        {"type": "action_trace_audit", "payload": {"action_trace": {
            "world_model_audit": {
                "possible_worlds": {"top_worlds": [{
                    "label": "world-a",
                    "key_assignments": {"p02": "werewolf"},
                    "why": ["claim:g1:0"],
                }]},
                "public_evidence_ids": ["claim:g1:0"],
            },
        }}},
    ]}])

    assert metrics["possible_world_evidence_covered_count"] == 0
    assert metrics["possible_world_evidence_coverage_rate"] == 0.0


def test_possible_world_hybrid_assignment_requires_exact_role_support() -> None:
    """未知或 hybrid 角色不能借笼统 good faction 声明获得证据覆盖。"""
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    metrics = compute_acceptance_audit_metrics([{
        "game_id": "g1",
        "players": {"p02": {"role": "hybrid"}},
        "events": [
        {"type": "speech", "payload": {
            "speaker": "p02", "text": "我是好人。",
            "claims": [{"type": "faction", "value": "good"}],
        }},
        {"type": "action_trace_audit", "payload": {"action_trace": {
            "world_model_audit": {
                "possible_worlds": {"top_worlds": [{
                    "label": "hybrid-world",
                    "key_assignments": {"p02": "hybrid"},
                    "why": ["claim:g1:0"],
                }]},
                "public_evidence_ids": ["claim:g1:0"],
            },
        }}},
    ]}])

    assert metrics["possible_world_evidence_covered_count"] == 0
    assert metrics["possible_world_evidence_coverage_rate"] == 0.0


def test_world_support_maps_faction_only_for_explicit_fixed_roles() -> None:
    from werewolf_agent.evaluation.balance_audit import _support_matches_world

    faction_only = {("p02", "faction:good")}

    assert _support_matches_world(
        faction_only, {"key_assignments": {"p02": "seer"}}
    ) is True
    assert _support_matches_world(
        faction_only, {"key_assignments": {"p02": "hybrid"}}
    ) is False
    assert _support_matches_world(
        faction_only, {"key_assignments": {"p02": "unknown_custom_role"}}
    ) is False


def test_possible_world_evidence_fails_closed_without_authoritative_index() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    metrics = compute_acceptance_audit_metrics([{
        "game_id": "g1",
        "players": {"p01": {"role": "villager"}, "p02": {"role": "werewolf"}},
        "events": [
        {"type": "speech", "payload": {"speaker": "p01", "text": "我怀疑p02。"}},
        {"type": "action_trace_audit", "payload": {"action_trace": {
            "world_model_audit": {"possible_worlds": {"top_worlds": [{
                "label": "w1", "key_assignments": {"p02": "werewolf"},
                "why": ["claim:g1:0"],
            }]}},
        }}},
    ]}])

    assert metrics["possible_world_evidence_covered_count"] == 0
    assert metrics["possible_world_evidence_coverage_rate"] == 0.0


def test_power_role_evidence_requires_explicit_legal_alternative_target() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    def event(comparison, *, retain=True):
        alternative = comparison.get("alternative_target")
        evidence = {
            "target_id": "p02",
            "target_evidence": {
                "selected_score": 5, "selected_signals": ["public_evidence"],
            },
            "target_comparison": {
                "selected_score": 5, "selected_signals": ["public_evidence"],
                "alternative_target": alternative,
                "alternative_score": 1 if alternative else None,
                "alternative_signals": ["weaker_evidence"] if alternative else [],
                "comparison_basis": "ranked public evidence score",
            },
            "friendly_fire_risk": {
                "status": "assessed", "targets": [], "basis": "无误伤目标",
            },
            "alternative_comparison": comparison,
        }
        if retain:
            evidence["retain_option"] = {
                "action": "no_action", "available": True,
                "required": False, "reason": "可选择不开枪",
            }
        return {"type": "action_trace_audit", "payload": {"action_trace": {
            "final_action_type": "hunter_shot",
            "power_role_evidence": evidence,
        }}}

    trace_events = [
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
    ]
    damage_events = [{"type": "player_died", "payload": {
        "player_id": "p02", "reason": "hunter_shot",
    }} for _ in trace_events]
    metrics = compute_acceptance_audit_metrics([{
        "game_id": "g-power-alternatives",
        "players": {"p01": {"role": "hunter"}, "p02": {"role": "werewolf"}},
        "deaths": [{
            "player_id": "p02", "reason": "hunter_shot",
        } for _ in trace_events],
        "events": [*trace_events, *damage_events],
    }])

    assert metrics["power_role_evidence_complete_count"] == 2
    assert metrics["power_role_evidence_completeness_rate"] == 0.4


def test_power_role_evidence_rejects_empty_or_partial_nested_objects() -> None:
    """空对象不能伪装成神职已比较风险、保留与替代方案。"""
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    base = {
        "target_id": "p02",
        "target_evidence": {
            "selected_score": 1, "selected_signals": ["公开查杀"],
        },
        "target_comparison": {
            "selected_score": 1, "selected_signals": ["公开查杀"],
            "alternative_target": "p03", "alternative_score": 0,
            "alternative_signals": ["no_structured_public_support"],
            "comparison_basis": "ranked public evidence score",
        },
        "friendly_fire_risk": {
            "status": "assessed", "targets": [], "basis": "无明显误伤证据",
        },
        "retain_option": {
            "action": "no_action", "available": True,
            "required": False, "reason": "可等待更强证据",
        },
        "alternative_comparison": {
            "legal_alternatives": ["p03"],
            "no_legal_alternative": False,
            "alternative_target": "p03",
        },
    }

    def event(evidence):
        return {"type": "action_trace_audit", "payload": {"action_trace": {
            "final_action_type": "use_poison",
            "power_role_evidence": evidence,
        }}}

    rows = [
        base,
        {**base, "friendly_fire_risk": {}},
        {**base, "retain_option": {"available": True}},
        {**base, "alternative_comparison": {"legal_alternatives": ["p03"]}},
    ]
    trace_events = [event(row) for row in rows]
    damage_events = [{"type": "player_died", "payload": {
        "player_id": "p02", "reason": "witch_poison",
    }} for _ in rows]
    metrics = compute_acceptance_audit_metrics([{
        "game_id": "g-power-nested",
        "players": {"p01": {"role": "witch"}, "p02": {"role": "werewolf"}},
        "deaths": [{
            "player_id": "p02", "reason": "witch_poison",
        } for _ in rows],
        "events": [*trace_events, *damage_events],
    }])

    assert metrics["power_role_evidence_complete_count"] == 1
    assert metrics["power_role_evidence_completeness_rate"] == 0.25


def test_reflection_coverage_requires_each_completed_game_audit() -> None:
    """一局完整审计不能掩盖另一已完成游戏完全没有反思事件。"""
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    clean = {
        "game_id": "g1", "winning_faction": "good", "players": {}, "events": [
            {"type": "reflection_complete", "payload": {
                "player_count": 0, "entries": [],
            }},
            {"type": "reflection_persistence_audit", "payload": {
                "expected_entry_count": 0,
                "persistence_complete": True,
                "rollback_complete": True,
                "entries": [],
            }},
        ],
    }
    missing = {"game_id": "g2", "winning_faction": "werewolf", "events": []}

    metrics = compute_acceptance_audit_metrics([clean, missing])

    assert metrics["reflection_completed_game_count"] == 2
    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False
    assert metrics["reflection_persisted_rejected_fact_count"] is None


def test_reflection_empty_transaction_is_never_supported() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    game = {
        "game_id": "g1", "winning_faction": "good", "players": {}, "events": [
            {"type": "reflection_complete", "payload": {
                "player_count": 0, "entries": [],
            }},
            {"type": "reflection_persistence_audit", "payload": {
                "expected_entry_count": 0,
                "persistence_complete": True,
                "rollback_complete": True,
                "entries": [],
            }},
        ],
    }

    metrics = compute_acceptance_audit_metrics([game])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False
    assert metrics["acceptance_projection_unsupported_reason"] == "missing_players"
    assert metrics["reflection_persisted_rejected_fact_count"] is None


def test_semantic_fallback_metric_requires_structured_fallback_kind() -> None:
    from werewolf_agent.evaluation.balance_audit import compute_acceptance_audit_metrics

    common = {
        "repairable": True, "success": False, "target_preserved": False,
        "introduced_claim_count": 0, "verified_claim_count": 0,
        "retained_verified_claim_count": 0, "generic_template_used": False,
    }
    projection = {"game_id": "g1", "players": {"p01": {"role": "villager"}}}
    legacy = compute_acceptance_audit_metrics([{**projection, "events": [
        *_semantic_pair(common, action_index=1),
    ]}])
    structured = compute_acceptance_audit_metrics([{**projection, "events": [
        *_semantic_pair(
            {**common, "fallback_kind": "task_specific"}, action_index=1
        ),
    ]}])

    assert legacy["semantic_repair_metrics_supported"] is False
    assert legacy["semantic_repair_generic_template_count"] is None
    assert structured["semantic_repair_metrics_supported"] is True
    assert structured["semantic_repair_generic_template_count"] == 0


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
