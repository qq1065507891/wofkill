# -*- coding: utf-8 -*-
"""
验证玩家动作跨轮修复与受控终退的连续证据链。

作者: Project contributors
修改日期: 2026-07-16
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


def test_terminal_fallback_preserves_structured_route_failure_reason() -> None:
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
        requested_reasoning_level=ReasoningLevel.HIGH,
        normalized_reasoning_status=ReasoningStatus.REQUESTED_UNCONFIRMED,
        reasoning_token_count=0,
        evidence_kind=EvidenceKind.NONE,
    ),))

    context.append_terminal_fallback("fallback_route_unavailable")

    assert context.terminal_failure_reason == "fallback_route_unavailable"
    assert context.attempts[-1].route_kind is RouteKind.SAFE_FALLBACK
