# -*- coding: utf-8 -*-
"""
验证 DecisionExecutionTrace V2 的尝试计数与旧版只读归一化。

作者: Project contributors
创建日期: 2026-07-15
"""

from __future__ import annotations

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
