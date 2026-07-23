# -*- coding: utf-8 -*-
"""
验证 DecisionExecutionTrace V2 的尝试计数与旧版只读归一化。

作者: Project contributors
创建日期: 2026-07-15
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from werewolf_agent.agents.trace_schemas import ActionTrace
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
from werewolf_agent.runtime.decision_outcomes import (
    AttemptCounts,
    derive_terminal_failure_code,
    normalize_decision_execution_trace,
    summarize_attempt_counts,
)


def _attempt(
    ordinal: int,
    route_kind: RouteKind,
    *,
    final: bool,
) -> AttemptExecutionRecord:
    success = final and route_kind is not RouteKind.SAFE_FALLBACK
    return AttemptExecutionRecord(
        opaque_request_id=OpaqueRequestId.new("game", "a1b2c3d4"),
        ordinal=ordinal,
        provider=(
            "backup" if route_kind is RouteKind.PROVIDER_FALLBACK else "primary"
        ),
        model="model-a",
        route_kind=route_kind,
        root_cause=(
            RootCause.NONE
            if success
            else RootCause.POLICY_REJECTION
            if route_kind is RouteKind.SAFE_FALLBACK
            else RootCause.TIMEOUT
        ),
        attempt_outcome=(
            AttemptOutcome.SUCCESS if success else AttemptOutcome.FAILURE
        ),
        requested_reasoning_level=ReasoningLevel.HIGH,
        normalized_reasoning_status=ReasoningStatus.REQUESTED_UNCONFIRMED,
        reasoning_token_count=0,
        evidence_kind=EvidenceKind.NONE,
    )


def _records(routes: tuple[RouteKind, ...]) -> tuple[AttemptExecutionRecord, ...]:
    return tuple(
        _attempt(index, route, final=index == len(routes))
        for index, route in enumerate(routes, 1)
    )


def test_v2_attempt_counters_follow_route_kinds() -> None:
    matrix = (
        ((RouteKind.PRIMARY,), (1, 0, 0)),
        ((RouteKind.PRIMARY, RouteKind.RETRY), (2, 1, 0)),
        ((RouteKind.PRIMARY, RouteKind.PROVIDER_FALLBACK), (2, 0, 1)),
        (
            (RouteKind.PRIMARY, RouteKind.RETRY, RouteKind.SAFE_FALLBACK),
            (3, 1, 0),
        ),
    )

    for routes, expected in matrix:
        counts = summarize_attempt_counts(_records(routes))
        assert (
            counts.attempt_count,
            counts.retry_count,
            counts.provider_fallback_count,
        ) == expected


def test_attempt_counts_timeout_default_rejects_boolean_and_negative_values() -> None:
    assert AttemptCounts(1, 0, 0).runtime_timeout_count == 0

    with pytest.raises(TypeError, match="runtime_timeout_count"):
        AttemptCounts(1, 0, 0, True)
    with pytest.raises(ValueError, match="runtime_timeout_count"):
        AttemptCounts(1, 0, 0, -1)


def test_runtime_timeout_count_uses_provider_attempts_only_for_records_and_mappings() -> None:
    attempts = (
        _attempt(1, RouteKind.PRIMARY, final=False),
        _attempt(2, RouteKind.RETRY, final=False),
        replace(
            _attempt(3, RouteKind.PROVIDER_FALLBACK, final=False),
            provider_attempted=False,
        ),
        replace(
            _attempt(4, RouteKind.SAFE_FALLBACK, final=True),
            root_cause=RootCause.TIMEOUT,
        ),
    )
    serialized = [
        {
            "opaque_request_id": attempt.opaque_request_id.value,
            "ordinal": attempt.ordinal,
            "provider": attempt.provider,
            "model": attempt.model,
            "route_kind": attempt.route_kind.value,
            "root_cause": attempt.root_cause.value,
            "attempt_outcome": attempt.attempt_outcome.value,
            "requested_reasoning_level": attempt.requested_reasoning_level.value,
            "normalized_reasoning_status": attempt.normalized_reasoning_status.value,
            "reasoning_token_count": attempt.reasoning_token_count,
            "evidence_kind": attempt.evidence_kind.value,
            "provider_attempted": attempt.provider_attempted,
        }
        for attempt in attempts
    ]
    # Task3 之前的序列化记录没有该字段，仍代表真实 provider 调用。
    del serialized[0]["provider_attempted"]
    del serialized[1]["provider_attempted"]

    assert summarize_attempt_counts(attempts).runtime_timeout_count == 2
    assert summarize_attempt_counts(serialized).runtime_timeout_count == 2


def test_runtime_timeout_count_defaults_legacy_objects_and_excludes_skipped_provider() -> None:
    legacy_timeout = SimpleNamespace(
        route_kind=RouteKind.PRIMARY,
        root_cause=RootCause.TIMEOUT,
    )
    skipped_timeout = SimpleNamespace(
        route_kind=RouteKind.RETRY,
        root_cause=RootCause.TIMEOUT,
        provider_attempted=False,
    )

    assert summarize_attempt_counts(
        [legacy_timeout, skipped_timeout]
    ).runtime_timeout_count == 1


def test_timeout_count_normalizes_legacy_traces_without_field_presence_drift() -> None:
    attempts = _records((RouteKind.PRIMARY, RouteKind.RETRY))
    legacy = {"execution_attempts": attempts}

    normalized = normalize_decision_execution_trace(legacy)
    trace = ActionTrace.model_validate(legacy)

    assert normalized["runtime_timeout_count"] == 1
    assert trace.runtime_timeout_count == 1
    assert "runtime_timeout_count" not in trace.model_dump(exclude_unset=True)
    assert "runtime_timeout_count" not in legacy
    assert normalize_decision_execution_trace({})["runtime_timeout_count"] == 0

    with pytest.raises(ValueError, match="runtime_timeout_count"):
        normalize_decision_execution_trace({
            "execution_attempts": attempts,
            "runtime_timeout_count": 0,
        })


def test_v1_trace_is_normalized_from_attempt_routes_without_mutation() -> None:
    attempts = _records((RouteKind.PRIMARY, RouteKind.PROVIDER_FALLBACK))
    legacy = {
        "execution_attempts": attempts,
        "retry_count": 1,
        "total_retry_count_until_success": 1,
    }

    normalized = normalize_decision_execution_trace(legacy)

    assert normalized["attempt_count"] == 2
    assert normalized["retry_count"] == 0
    assert normalized["provider_fallback_count"] == 1
    assert normalized["normalized_from_schema_version"] == "1"
    assert legacy["retry_count"] == 1
    assert "normalized_from_schema_version" not in legacy


def test_action_trace_writes_only_v2_counter_fields() -> None:
    trace = ActionTrace(
        attempt_count=2,
        retry_count=0,
        provider_fallback_count=1,
        generated_by="provider_fallback",
    )

    payload = trace.model_dump()

    assert payload["attempt_count"] == 2
    assert payload["provider_fallback_count"] == 1
    assert payload["generated_by"] == "provider_fallback"
    assert payload["terminal_failure_code"] is None
    assert "total_retry_count_until_success" not in payload


def test_terminal_failure_code_is_stable_across_v2_normalization() -> None:
    attempts = _records((RouteKind.PRIMARY, RouteKind.SAFE_FALLBACK))
    cases = (
        ("illegal_action", "illegal_action"),
        ("empty_response", "empty_response"),
        ("invalid_output", "invalid_output"),
        ("provider_error", "provider_error"),
        ("prompt player p01 role", "unknown"),
    )

    for candidate, expected in cases:
        code = derive_terminal_failure_code(attempts, candidate)
        trace = {
            "execution_attempts": attempts,
            "attempt_count": 2,
            "retry_count": 0,
            "provider_fallback_count": 0,
            "generated_by": "terminal_fallback",
            "terminal_failure_code": code,
            "structured_failure_reason": candidate,
        }
        normalized = normalize_decision_execution_trace(trace)

        assert code == expected
        assert normalized["terminal_failure_code"] == expected
        assert normalize_decision_execution_trace(normalized) == normalized


def test_v1_terminal_trace_uses_attempt_root_cause_without_rewrite() -> None:
    attempts = _records((RouteKind.PRIMARY, RouteKind.SAFE_FALLBACK))
    legacy = {"execution_attempts": attempts, "retry_count": 7}

    normalized = normalize_decision_execution_trace(legacy)

    assert normalized["terminal_failure_code"] == "policy_rejection"
    assert normalized["normalized_from_schema_version"] == "1"
    assert legacy == {"execution_attempts": attempts, "retry_count": 7}
