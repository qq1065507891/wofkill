# -*- coding: utf-8 -*-
"""
验证玩家动作跨轮修复与受控终退的连续证据链。

作者: Project contributors
修改日期: 2026-07-24
"""

import pytest


@pytest.mark.parametrize("failure_kind", ["invalid_json", "semantic_invalid"])
def test_outer_repair_keeps_one_contiguous_chain_and_translates_repair_success(failure_kind):
    from werewolf_agent.model_gateway.execution_records import (
        AttemptExecutionRecord, AttemptOutcome, EvidenceKind, ReasoningLevel,
        ReasoningStatus, RootCause, RouteKind,
    )
    from werewolf_agent.model_gateway.generation_attempt_context import GenerationAttemptContext
    from werewolf_agent.runtime.decision_outcomes import translate_decision_outcome

    context = GenerationAttemptContext("p01x")
    first = AttemptExecutionRecord(
        opaque_request_id=context.opaque_request_id, ordinal=1,
        provider="openai", model="m", route_kind=RouteKind.PRIMARY,
        root_cause=RootCause.NONE, attempt_outcome=AttemptOutcome.SUCCESS,
        requested_reasoning_level=ReasoningLevel.HIGH,
        normalized_reasoning_status=ReasoningStatus.CONFIRMED,
        reasoning_token_count=3, evidence_kind=EvidenceKind.TOKEN_COUNT,
    )
    context.accept((first,))
    context.reject_latest_output()
    final = AttemptExecutionRecord(
        opaque_request_id=context.opaque_request_id, ordinal=2,
        provider="openai", model="m", route_kind=context.next_route_kind,
        root_cause=RootCause.NONE, attempt_outcome=AttemptOutcome.SUCCESS,
        requested_reasoning_level=ReasoningLevel.HIGH,
        normalized_reasoning_status=ReasoningStatus.CONFIRMED,
        reasoning_token_count=2, evidence_kind=EvidenceKind.TOKEN_COUNT,
    )
    context.accept((*context.attempts, final))
    translated = translate_decision_outcome(context.attempts)
    assert [attempt.ordinal for attempt in context.attempts] == [1, 2]
    assert len({attempt.opaque_request_id.value for attempt in context.attempts}) == 1
    assert translated.outcome.value == "repaired_success"
    assert sum(attempt.reasoning_token_count for attempt in context.attempts) == 5


@pytest.mark.parametrize(
    (
        "requested_level",
        "source_status",
        "expected_terminal_status",
        "expected_terminal_evidence",
    ),
    [
        (
            "high",
            "requested_unconfirmed",
            "fallback_disabled",
            "fallback_disabled",
        ),
        ("none", "not_requested", "not_requested", "none"),
    ],
)
def test_terminal_fallback_marks_deterministic_boundary_as_provider_skipped(
    requested_level: str,
    source_status: str,
    expected_terminal_status: str,
    expected_terminal_evidence: str,
) -> None:
    from werewolf_agent.model_gateway.execution_records import (
        AttemptExecutionRecord, AttemptOutcome, EvidenceKind, ReasoningLevel,
        ReasoningStatus, RootCause, RouteKind,
    )
    from werewolf_agent.model_gateway.generation_attempt_context import GenerationAttemptContext

    context = GenerationAttemptContext("p01x")
    context.accept((AttemptExecutionRecord(
        opaque_request_id=context.opaque_request_id,
        ordinal=1,
        provider="primary",
        model="model-a",
        route_kind=RouteKind.PRIMARY,
        root_cause=RootCause.PROVIDER_ERROR,
        attempt_outcome=AttemptOutcome.FAILURE,
        requested_reasoning_level=ReasoningLevel(requested_level),
        normalized_reasoning_status=ReasoningStatus(source_status),
        reasoning_token_count=0,
        evidence_kind=EvidenceKind.NONE,
    ),))

    context.append_terminal_fallback("fallback_route_unavailable")

    assert context.terminal_failure_reason == "fallback_route_unavailable"
    assert len(context.attempts) == 2
    actual, terminal = context.attempts
    assert actual.provider_attempted is True
    assert terminal.provider_attempted is False
    assert terminal.opaque_request_id == actual.opaque_request_id
    assert (terminal.provider, terminal.model) == (actual.provider, actual.model)
    assert terminal.route_kind is RouteKind.SAFE_FALLBACK
    assert terminal.root_cause is RootCause.INVALID_OUTPUT
    assert terminal.attempt_outcome is AttemptOutcome.FAILURE
    assert terminal.reasoning_token_count == 0
    assert terminal.normalized_reasoning_status is ReasoningStatus(
        expected_terminal_status
    )
    assert terminal.evidence_kind is EvidenceKind(expected_terminal_evidence)
