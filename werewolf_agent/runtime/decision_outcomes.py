# -*- coding: utf-8 -*-
"""
把逐次执行事实翻译为唯一、互斥的最终决策结果。

作者: Project contributors
创建日期: 2026-07-13
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from werewolf_agent.model_gateway.execution_records import (
    AttemptExecutionRecord,
    AttemptOutcome,
    RouteKind,
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
    """最终分类及完整可追溯尝试链。"""

    outcome: DecisionOutcome
    retry_count: int
    attempts: tuple[AttemptExecutionRecord, ...]

    @property
    def final_attempt(self) -> AttemptExecutionRecord:
        """返回产生最终结果的尝试。"""
        return self.attempts[-1]


def translate_decision_outcome(
    attempts: tuple[AttemptExecutionRecord, ...],
) -> TranslatedDecisionOutcome:
    """仅按强类型路线与尝试事实翻译结果，不解析自由文本。"""
    if not attempts:
        raise ValueError("at least one attempt is required")
    if tuple(item.ordinal for item in attempts) != tuple(range(1, len(attempts) + 1)):
        raise ValueError("attempt ordinals must be contiguous and start at 1")
    if attempts[0].route_kind is not RouteKind.PRIMARY:
        raise ValueError("attempt sequence must start with a primary route")
    terminal_positions = [
        index for index, item in enumerate(attempts)
        if item.route_kind is RouteKind.SAFE_FALLBACK
    ]
    if terminal_positions and terminal_positions != [len(attempts) - 1]:
        raise ValueError("terminal safe fallback must be the final attempt")
    if any(item.attempt_outcome is not AttemptOutcome.FAILURE for item in attempts[:-1]):
        raise ValueError("only failed attempts may precede the final attempt")
    final = attempts[-1]
    if final.attempt_outcome is AttemptOutcome.FAILURE:
        raise ValueError("attempt sequence has no successful terminal outcome")

    route_outcomes = {
        RouteKind.PRIMARY: DecisionOutcome.DIRECT_SUCCESS,
        RouteKind.RETRY: DecisionOutcome.RETRY_SUCCESS,
        RouteKind.REPAIR: DecisionOutcome.REPAIRED_SUCCESS,
        RouteKind.PROVIDER_FALLBACK: DecisionOutcome.PROVIDER_FALLBACK_SUCCESS,
        RouteKind.SAFE_FALLBACK: DecisionOutcome.TERMINAL_FALLBACK,
    }
    return TranslatedDecisionOutcome(
        outcome=route_outcomes[final.route_kind],
        retry_count=len(attempts) - 1,
        attempts=attempts,
    )


__all__ = [
    "DecisionOutcome",
    "TranslatedDecisionOutcome",
    "translate_decision_outcome",
]
