# -*- coding: utf-8 -*-
"""
权威决策结果分类、最终来源归因、V1 反序列化与非法序列测试。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-16
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
    DecisionGeneratedBy,
    DecisionOutcome,
    normalize_decision_execution_trace,
    translate_decision_outcome,
    translate_serialized_decision_outcome,
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


def _serialized_attempt(
    attempt: AttemptExecutionRecord,
) -> dict[str, object]:
    return {
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
    }


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
            0,
        ),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.PROVIDER_ERROR),
                _attempt(2, RouteKind.PROVIDER_FALLBACK, AttemptOutcome.SUCCESS),
            ),
            DecisionOutcome.PROVIDER_FALLBACK_SUCCESS,
            0,
        ),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.POLICY_REJECTION),
                _attempt(2, RouteKind.SAFE_FALLBACK, AttemptOutcome.FAILURE, cause=RootCause.POLICY_REJECTION),
            ),
            DecisionOutcome.TERMINAL_FALLBACK,
            0,
        ),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT),
                _attempt(2, RouteKind.PROVIDER_FALLBACK, AttemptOutcome.FAILURE, cause=RootCause.PROVIDER_ERROR),
                replace(
                    _attempt(3, RouteKind.RETRY, AttemptOutcome.SUCCESS),
                    provider="backup",
                ),
            ),
            DecisionOutcome.PROVIDER_FALLBACK_SUCCESS,
            1,
        ),
    ],
)
def test_retry_semantics_table(attempts, expected, retries) -> None:
    result = translate_decision_outcome(attempts)
    assert (result.outcome, result.retry_count) == (expected, retries)
    assert result.attempt_count == len(attempts)
    assert result.provider_fallback_count == sum(
        item.route_kind is RouteKind.PROVIDER_FALLBACK for item in attempts
    )


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


def test_fallback_route_unavailable_remains_a_stable_terminal_classification() -> None:
    attempts = (
        _attempt(
            1, RouteKind.PRIMARY, AttemptOutcome.FAILURE,
            cause=RootCause.PROVIDER_ERROR,
        ),
        _attempt(
            2, RouteKind.SAFE_FALLBACK, AttemptOutcome.FAILURE,
            cause=RootCause.POLICY_REJECTION,
        ),
    )

    result = translate_decision_outcome(
        attempts,
        structured_failure_reason="fallback_route_unavailable",
    )

    assert result.terminal_failure_code == "fallback_route_unavailable"


@pytest.mark.parametrize(
    ("attempts", "expected_generated_by", "expected_outcome"),
    [
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.INVALID_OUTPUT),
                _attempt(2, RouteKind.REPAIR, AttemptOutcome.FAILURE, cause=RootCause.INVALID_OUTPUT),
                _attempt(3, RouteKind.PROVIDER_FALLBACK, AttemptOutcome.SUCCESS),
            ),
            DecisionGeneratedBy.PROVIDER_FALLBACK,
            DecisionOutcome.REPAIRED_SUCCESS,
        ),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.INVALID_OUTPUT),
                _attempt(2, RouteKind.REPAIR, AttemptOutcome.FAILURE, cause=RootCause.INVALID_OUTPUT),
                _attempt(3, RouteKind.PROVIDER_FALLBACK, AttemptOutcome.FAILURE, cause=RootCause.PROVIDER_ERROR),
                replace(_attempt(4, RouteKind.RETRY, AttemptOutcome.SUCCESS), provider="backup"),
            ),
            DecisionGeneratedBy.PROVIDER_FALLBACK,
            DecisionOutcome.REPAIRED_SUCCESS,
        ),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.PROVIDER_ERROR),
                _attempt(2, RouteKind.PROVIDER_FALLBACK, AttemptOutcome.FAILURE, cause=RootCause.INVALID_OUTPUT),
                _attempt(3, RouteKind.REPAIR, AttemptOutcome.SUCCESS),
            ),
            DecisionGeneratedBy.REPAIR,
            DecisionOutcome.REPAIRED_SUCCESS,
        ),
        (
            (
                _attempt(1, RouteKind.PRIMARY, AttemptOutcome.FAILURE, cause=RootCause.TIMEOUT),
                _attempt(2, RouteKind.RETRY, AttemptOutcome.SUCCESS),
            ),
            DecisionGeneratedBy.MODEL,
            DecisionOutcome.RETRY_SUCCESS,
        ),
    ],
)
def test_generated_by_tracks_final_content_source_for_runtime_and_v1(
    attempts: tuple[AttemptExecutionRecord, ...],
    expected_generated_by: DecisionGeneratedBy,
    expected_outcome: DecisionOutcome,
) -> None:
    runtime = translate_decision_outcome(attempts)
    serialized = translate_serialized_decision_outcome(
        [_serialized_attempt(item) for item in attempts]
    )
    normalized = normalize_decision_execution_trace(
        {"execution_attempts": [_serialized_attempt(item) for item in attempts]}
    )

    assert runtime.generated_by is expected_generated_by
    assert serialized.generated_by is expected_generated_by
    assert normalized["generated_by"] == expected_generated_by.value
    assert runtime.outcome is expected_outcome
    assert serialized.outcome is expected_outcome


@pytest.mark.parametrize(
    "ordinal",
    [
        True,
        False,
        1.0,
        1.5,
        0,
        -1,
        "",
        "0",
        "-1",
        "+1",
        "01",
        " 1",
        "1 ",
        "1.0",
        "１",
        "١",
        None,
    ],
)
def test_v1_ordinal_rejects_noncanonical_values(ordinal: object) -> None:
    attempt = _attempt(1, RouteKind.PRIMARY, AttemptOutcome.SUCCESS)
    payload = _serialized_attempt(attempt)
    payload["ordinal"] = ordinal

    with pytest.raises(ValueError, match="ordinal"):
        translate_serialized_decision_outcome([payload])


@pytest.mark.parametrize("ordinal", [1, "1"])
def test_v1_ordinal_accepts_exact_int_or_canonical_ascii_string(
    ordinal: int | str,
) -> None:
    attempt = _attempt(1, RouteKind.PRIMARY, AttemptOutcome.SUCCESS)
    payload = _serialized_attempt(attempt)
    payload["ordinal"] = ordinal

    translated = translate_serialized_decision_outcome([payload])

    assert translated.final_attempt.ordinal == 1


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
