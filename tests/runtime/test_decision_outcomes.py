# -*- coding: utf-8 -*-
"""
权威决策结果分类、重试语义与非法序列测试。

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
from werewolf_agent.runtime.decision_outcomes import (
    DecisionOutcome,
    translate_decision_outcome,
)


def _attempt(
    ordinal: int,
    route_kind: RouteKind,
    outcome: AttemptOutcome,
    *,
    cause: RootCause = RootCause.NONE,
    reasoning_status: str = "requested_unconfirmed",
    reasoning_tokens: int = 0,
) -> AttemptExecutionRecord:
    return AttemptExecutionRecord(
        opaque_request_id=OpaqueRequestId.new("game", "abcd1234"),
        ordinal=ordinal,
        provider="primary" if route_kind is not RouteKind.PROVIDER_FALLBACK else "backup",
        model="model-a",
        route_kind=route_kind,
        root_cause=cause,
        attempt_outcome=outcome,
        requested_reasoning_level=ReasoningLevel.HIGH,
        normalized_reasoning_status=ReasoningStatus(reasoning_status),
        reasoning_token_count=reasoning_tokens,
        evidence_kind=(EvidenceKind.AUTHORITATIVE_PROVIDER_EXECUTION if reasoning_status == "confirmed" else EvidenceKind.NONE),
    )


def test_outcome_taxonomy_is_mutually_exclusive() -> None:
    assert set(AttemptOutcome) == {AttemptOutcome.SUCCESS, AttemptOutcome.FAILURE}
    assert not ({item.value for item in AttemptOutcome} & {item.value for item in DecisionOutcome})
    assert set(RouteKind) == {
        RouteKind.PRIMARY,
        RouteKind.RETRY,
        RouteKind.PROVIDER_FALLBACK,
        RouteKind.REPAIR,
        RouteKind.SAFE_FALLBACK,
    }
    with pytest.raises(ValueError, match="failed attempt requires"):
        _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE)
    with pytest.raises(ValueError, match="successful attempt cannot"):
        _attempt(
            1,
            RouteKind.PRIMARY,
            AttemptOutcome.SUCCESS,
            cause=RootCause.PROVIDER_ERROR,
        )


@pytest.mark.parametrize(
    ("attempts", "expected", "retries"),
    [
        ((_attempt(1, RouteKind.PRIMARY, AttemptOutcome.SUCCESS),), DecisionOutcome.DIRECT_SUCCESS, 0),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT),
                _attempt(2, RouteKind.RETRY, AttemptOutcome.SUCCESS),
            ),
            DecisionOutcome.RETRY_SUCCESS,
            1,
        ),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.INVALID_OUTPUT),
                _attempt(2, RouteKind.REPAIR, AttemptOutcome.SUCCESS),
            ),
            DecisionOutcome.REPAIRED_SUCCESS,
            1,
        ),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.PROVIDER_ERROR),
                _attempt(2, RouteKind.PROVIDER_FALLBACK, AttemptOutcome.SUCCESS),
            ),
            DecisionOutcome.PROVIDER_FALLBACK_SUCCESS,
            1,
        ),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.POLICY_REJECTION),
                _attempt(2, RouteKind.SAFE_FALLBACK, AttemptOutcome.FAILURE, cause=RootCause.POLICY_REJECTION),
            ),
            DecisionOutcome.TERMINAL_FALLBACK,
            1,
        ),
    ],
)
def test_retry_semantics_table(attempts, expected, retries) -> None:
    result = translate_decision_outcome(attempts)
    assert (result.outcome, result.retry_count) == (expected, retries)


def test_provider_fallback_preserves_reasoning_evidence() -> None:
    attempts = (
        _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT),
        _attempt(
            2,
            RouteKind.PROVIDER_FALLBACK,
            AttemptOutcome.SUCCESS,
            reasoning_status="confirmed",
            reasoning_tokens=23,
        ),
    )
    result = translate_decision_outcome(attempts)
    assert result.final_attempt.normalized_reasoning_status == "confirmed"
    assert result.final_attempt.reasoning_token_count == 23
    assert result.attempts[0].root_cause is RootCause.TIMEOUT


def test_illegal_attempt_sequences_are_rejected() -> None:
    with pytest.raises(ValueError, match="start with a primary"):
        translate_decision_outcome((_attempt(1, RouteKind.RETRY, AttemptOutcome.SUCCESS),))
    with pytest.raises(ValueError, match="cannot be repeated"):
        translate_decision_outcome(
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT),
                _attempt(2, RouteKind.PRIMARY, AttemptOutcome.SUCCESS),
            )
        )

    with pytest.raises(ValueError, match="must switch"):
        translate_decision_outcome(
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT),
                replace(
                    _attempt(
                        2,
                        RouteKind.PROVIDER_FALLBACK,
                        AttemptOutcome.FAILURE,
                        cause=RootCause.PROVIDER_ERROR,
                    ),
                    provider="primary",
                ),
                _attempt(3, RouteKind.RETRY, AttemptOutcome.SUCCESS),
            )
        )
    with pytest.raises(ValueError, match="contiguous"):
        translate_decision_outcome(
            (_attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT), _attempt(3, RouteKind.RETRY, AttemptOutcome.SUCCESS))
        )
    with pytest.raises(ValueError, match="terminal"):
        translate_decision_outcome(
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT),
                _attempt(2, RouteKind.SAFE_FALLBACK, AttemptOutcome.FAILURE, cause=RootCause.POLICY_REJECTION),
                _attempt(3, RouteKind.RETRY, AttemptOutcome.SUCCESS),
            )
        )
    with pytest.raises(ValueError, match="failed attempt"):
        translate_decision_outcome(
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT),
                _attempt(2, RouteKind.SAFE_FALLBACK, AttemptOutcome.SUCCESS),
            )
        )
