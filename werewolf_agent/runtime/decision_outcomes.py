# -*- coding: utf-8 -*-
"""
把逐次模型执行记录翻译为唯一、互斥的最终决策结果。

作者: Project contributors
创建日期: 2026-07-13

使用示例:
    >>> translate_decision_outcome((attempt,)).outcome
    <DecisionOutcome.DIRECT_SUCCESS: 'direct_success'>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from werewolf_agent.model_gateway.execution_records import (
    AttemptExecutionRecord,
    AttemptOutcome,
)


class DecisionOutcome(str, Enum):
    """一次完整决策调用的互斥最终结果。"""

    DIRECT_SUCCESS = "direct_success"
    RETRY_SUCCESS = "retry_success"
    REPAIRED_SUCCESS = "repaired_success"
    PROVIDER_FALLBACK_SUCCESS = "provider_fallback_success"
    TERMINAL_FALLBACK = "terminal_fallback"


@dataclass(frozen=True)
class TranslatedDecisionOutcome:
    """最终分类及由尝试序列推导出的重试次数。"""

    outcome: DecisionOutcome
    retry_count: int


def translate_decision_outcome(
    attempts: tuple[AttemptExecutionRecord, ...],
) -> TranslatedDecisionOutcome:
    """按强类型尝试结果翻译最终分类，不解析自由文本。"""
    if not attempts:
        raise ValueError("at least one attempt is required")
    expected_numbers = tuple(range(1, len(attempts) + 1))
    if tuple(item.attempt_number for item in attempts) != expected_numbers:
        raise ValueError("attempt numbers must be contiguous and start at 1")
    if any(item.outcome is not AttemptOutcome.RETRYABLE_FAILURE for item in attempts[:-1]):
        raise ValueError("only retryable failures may precede the final attempt")

    final = attempts[-1].outcome
    retry_count = len(attempts) - 1
    outcome_map = {
        AttemptOutcome.SUCCESS: (
            DecisionOutcome.RETRY_SUCCESS if retry_count else DecisionOutcome.DIRECT_SUCCESS
        ),
        AttemptOutcome.REPAIRED_SUCCESS: DecisionOutcome.REPAIRED_SUCCESS,
        AttemptOutcome.PROVIDER_FALLBACK_SUCCESS: DecisionOutcome.PROVIDER_FALLBACK_SUCCESS,
        AttemptOutcome.TERMINAL_FALLBACK: DecisionOutcome.TERMINAL_FALLBACK,
    }
    if final is AttemptOutcome.RETRYABLE_FAILURE:
        raise ValueError("attempt sequence has no terminal outcome")
    return TranslatedDecisionOutcome(outcome=outcome_map[final], retry_count=retry_count)


__all__ = [
    "DecisionOutcome",
    "TranslatedDecisionOutcome",
    "translate_decision_outcome",
]
