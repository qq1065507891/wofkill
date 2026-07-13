# -*- coding: utf-8 -*-
"""
把逐次执行事实翻译为唯一、互斥的最终决策结果。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-13
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from werewolf_agent.model_gateway.execution_records import (
    AttemptOutcome,
    EvidenceKind,
    ReasoningLevel,
    ReasoningStatus,
    RootCause,
    RouteKind,
)


class DecisionAttempt(Protocol):
    """供运行时对象与脱敏 JSON 投影共同使用的最小尝试协议。"""

    opaque_request_id: Any
    ordinal: int
    provider: str
    model: str
    route_kind: RouteKind
    root_cause: RootCause
    attempt_outcome: AttemptOutcome
    requested_reasoning_level: ReasoningLevel
    normalized_reasoning_status: ReasoningStatus
    reasoning_token_count: int
    evidence_kind: EvidenceKind


@dataclass(frozen=True)
class _SerializedDecisionAttempt:
    """把 JSON 白名单字段恢复为强类型枚举，不恢复供应商请求 ID。"""

    opaque_request_id: str
    ordinal: int
    provider: str
    model: str
    route_kind: RouteKind
    root_cause: RootCause
    attempt_outcome: AttemptOutcome
    requested_reasoning_level: ReasoningLevel
    normalized_reasoning_status: ReasoningStatus
    reasoning_token_count: int
    evidence_kind: EvidenceKind


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
    attempts: tuple[DecisionAttempt, ...]

    @property
    def final_attempt(self) -> DecisionAttempt:
        """返回产生最终结果的尝试。"""
        return self.attempts[-1]


def translate_decision_outcome(
    attempts: tuple[DecisionAttempt, ...],
) -> TranslatedDecisionOutcome:
    """仅按强类型路线与尝试事实翻译结果，不解析自由文本。"""
    if not attempts:
        raise ValueError("at least one attempt is required")
    if tuple(item.ordinal for item in attempts) != tuple(range(1, len(attempts) + 1)):
        raise ValueError("attempt ordinals must be contiguous and start at 1")
    if attempts[0].route_kind is not RouteKind.PRIMARY:
        raise ValueError("attempt sequence must start with a primary route")
    if any(item.route_kind is RouteKind.PRIMARY for item in attempts[1:]):
        raise ValueError("primary route cannot be repeated")
    if any(item.opaque_request_id != attempts[0].opaque_request_id for item in attempts[1:]):
        raise ValueError("all attempts must share one opaque request id")
    terminal_positions = [
        index for index, item in enumerate(attempts)
        if item.route_kind is RouteKind.SAFE_FALLBACK
    ]
    if terminal_positions and terminal_positions != [len(attempts) - 1]:
        raise ValueError("terminal safe fallback must be the final attempt")
    if any(item.attempt_outcome is not AttemptOutcome.FAILURE for item in attempts[:-1]):
        raise ValueError("only failed attempts may precede the final attempt")
    final = attempts[-1]
    for index, item in enumerate(attempts[1:], start=1):
        if item.route_kind is not RouteKind.PROVIDER_FALLBACK:
            continue
        previous = attempts[index - 1]
        if (item.provider, item.model) == (previous.provider, previous.model):
            raise ValueError("provider fallback must switch provider or model")
    if final.route_kind is RouteKind.SAFE_FALLBACK:
        if final.attempt_outcome is not AttemptOutcome.FAILURE:
            raise ValueError("terminal safe fallback must record a failed attempt")
    elif final.attempt_outcome is AttemptOutcome.FAILURE:
        raise ValueError("attempt sequence has no successful terminal outcome")

    route_history = {item.route_kind for item in attempts}
    if RouteKind.SAFE_FALLBACK in route_history:
        outcome = DecisionOutcome.TERMINAL_FALLBACK
    elif RouteKind.REPAIR in route_history:
        outcome = DecisionOutcome.REPAIRED_SUCCESS
    elif RouteKind.PROVIDER_FALLBACK in route_history:
        outcome = DecisionOutcome.PROVIDER_FALLBACK_SUCCESS
    elif RouteKind.RETRY in route_history:
        outcome = DecisionOutcome.RETRY_SUCCESS
    else:
        outcome = DecisionOutcome.DIRECT_SUCCESS
    return TranslatedDecisionOutcome(
        outcome=outcome,
        retry_count=len(attempts) - 1,
        attempts=attempts,
    )


def translate_serialized_decision_outcome(
    attempts: Sequence[Mapping[str, Any]],
) -> TranslatedDecisionOutcome:
    """规范化脱敏 JSON 尝试后复用唯一 translator，拒绝未知枚举。"""
    normalized = tuple(_serialized_attempt(item) for item in attempts)
    return translate_decision_outcome(normalized)


def _serialized_attempt(payload: Mapping[str, Any]) -> _SerializedDecisionAttempt:
    request_id = payload.get("opaque_request_id")
    if isinstance(request_id, Mapping):
        request_id = request_id.get("value")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("serialized attempt requires an opaque request id")
    return _SerializedDecisionAttempt(
        opaque_request_id=request_id,
        ordinal=int(payload["ordinal"]),
        provider=str(payload["provider"]),
        model=str(payload["model"]),
        route_kind=RouteKind(payload["route_kind"]),
        root_cause=RootCause(payload["root_cause"]),
        attempt_outcome=AttemptOutcome(payload["attempt_outcome"]),
        requested_reasoning_level=ReasoningLevel(
            payload["requested_reasoning_level"]
        ),
        normalized_reasoning_status=ReasoningStatus(
            payload["normalized_reasoning_status"]
        ),
        reasoning_token_count=int(payload.get("reasoning_token_count") or 0),
        evidence_kind=EvidenceKind(payload["evidence_kind"]),
    )


__all__ = [
    "DecisionOutcome",
    "DecisionAttempt",
    "TranslatedDecisionOutcome",
    "translate_decision_outcome",
    "translate_serialized_decision_outcome",
]
