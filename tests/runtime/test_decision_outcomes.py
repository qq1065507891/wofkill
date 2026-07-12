# -*- coding: utf-8 -*-
"""
权威决策结果分类与重试语义测试。

作者: Project contributors
创建日期: 2026-07-13
"""

from __future__ import annotations

from werewolf_agent.model_gateway.execution_records import (
    AttemptExecutionRecord,
    AttemptOutcome,
    RootCause,
)
from werewolf_agent.runtime.decision_outcomes import (
    DecisionOutcome,
    translate_decision_outcome,
)


def _attempt(number: int, outcome: AttemptOutcome) -> AttemptExecutionRecord:
    root_cause = RootCause.NONE if outcome.is_success else RootCause.PROVIDER_ERROR
    return AttemptExecutionRecord(
        attempt_number=number,
        provider="primary",
        model="model-a",
        outcome=outcome,
        root_cause=root_cause,
    )


def test_outcome_taxonomy_is_mutually_exclusive() -> None:
    values = [outcome.value for outcome in DecisionOutcome]

    assert len(values) == len(set(values))
    assert {
        DecisionOutcome.DIRECT_SUCCESS,
        DecisionOutcome.RETRY_SUCCESS,
        DecisionOutcome.REPAIRED_SUCCESS,
        DecisionOutcome.PROVIDER_FALLBACK_SUCCESS,
        DecisionOutcome.TERMINAL_FALLBACK,
    } == set(DecisionOutcome)


def test_retry_semantics_for_repair_provider_fallback_and_terminal_fallback() -> None:
    repaired = translate_decision_outcome(
        (_attempt(1, AttemptOutcome.RETRYABLE_FAILURE), _attempt(2, AttemptOutcome.REPAIRED_SUCCESS))
    )
    provider_fallback = translate_decision_outcome(
        (
            _attempt(1, AttemptOutcome.RETRYABLE_FAILURE),
            _attempt(2, AttemptOutcome.PROVIDER_FALLBACK_SUCCESS),
        )
    )
    terminal = translate_decision_outcome(
        (
            _attempt(1, AttemptOutcome.RETRYABLE_FAILURE),
            _attempt(2, AttemptOutcome.TERMINAL_FALLBACK),
        )
    )

    assert (repaired.outcome, repaired.retry_count) == (DecisionOutcome.REPAIRED_SUCCESS, 1)
    assert (provider_fallback.outcome, provider_fallback.retry_count) == (
        DecisionOutcome.PROVIDER_FALLBACK_SUCCESS,
        1,
    )
    assert (terminal.outcome, terminal.retry_count) == (DecisionOutcome.TERMINAL_FALLBACK, 1)
