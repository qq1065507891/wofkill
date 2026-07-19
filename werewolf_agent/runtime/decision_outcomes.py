# -*- coding: utf-8 -*-
"""
把逐次执行事实翻译为唯一、互斥的最终决策结果。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-19
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


class DecisionGeneratedBy(str, Enum):
    """最终动作内容的稳定生成来源。"""

    MODEL = "model"
    REPAIR = "repair"
    PROVIDER_FALLBACK = "provider_fallback"
    TERMINAL_FALLBACK = "terminal_fallback"


STABLE_TERMINAL_FAILURE_CODES = frozenset({
    "empty_response",
    "fallback_route_unavailable",
    "illegal_action",
    "invalid_output",
    "invalid_tool_arguments",
    "missing_tool_call",
    "model_generation_failed",
    "parse_error",
    "policy_rejection",
    "provider_error",
    "schema_validation",
    "semantic_claim_retention",
    "speech_quality",
    "structured_output_unsupported",
    "timeout",
    "truncated_json",
    "unknown",
    "vote_quality",
})


@dataclass(frozen=True)
class AttemptCounts:
    """仅由强类型路由推导的 V2 尝试计数。"""

    attempt_count: int
    retry_count: int
    provider_fallback_count: int


@dataclass(frozen=True)
class TranslatedDecisionOutcome:
    """最终分类及完整可追溯尝试链。"""

    outcome: DecisionOutcome
    retry_count: int
    attempt_count: int
    provider_fallback_count: int
    generated_by: DecisionGeneratedBy
    terminal_failure_code: str | None
    attempts: tuple[DecisionAttempt, ...]

    @property
    def final_attempt(self) -> DecisionAttempt:
        """返回产生最终结果的尝试。"""
        return self.attempts[-1]


def translate_decision_outcome(
    attempts: tuple[DecisionAttempt, ...],
    *,
    structured_failure_reason: str | None = None,
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
    counts = summarize_attempt_counts(attempts)
    return TranslatedDecisionOutcome(
        outcome=outcome,
        retry_count=counts.retry_count,
        attempt_count=counts.attempt_count,
        provider_fallback_count=counts.provider_fallback_count,
        generated_by=derive_generated_by(attempts),
        terminal_failure_code=derive_terminal_failure_code(
            attempts,
            structured_failure_reason,
        ),
        attempts=attempts,
    )


def summarize_attempt_counts(
    attempts: Sequence[DecisionAttempt | Mapping[str, Any]],
) -> AttemptCounts:
    """按集中定义的 route_kind 语义计算尝试、重试和供应商回退次数。"""
    route_kinds = tuple(_route_kind(item) for item in attempts)
    return AttemptCounts(
        attempt_count=len(route_kinds),
        retry_count=sum(route is RouteKind.RETRY for route in route_kinds),
        provider_fallback_count=sum(
            route is RouteKind.PROVIDER_FALLBACK for route in route_kinds
        ),
    )


def normalize_terminal_failure_code(value: Any) -> str:
    """把终态失败原因收敛到封闭稳定码集合，未知值不回显。"""
    return value if isinstance(value, str) and value in STABLE_TERMINAL_FAILURE_CODES else "unknown"


def derive_terminal_failure_code(
    attempts: Sequence[DecisionAttempt | Mapping[str, Any]],
    structured_failure_reason: str | None = None,
) -> str | None:
    """按统一优先级从终态 attempt 与结构化失败原因推导稳定码。"""
    if not attempts or _route_kind(attempts[-1]) is not RouteKind.SAFE_FALLBACK:
        return None
    if structured_failure_reason is not None:
        return normalize_terminal_failure_code(structured_failure_reason)
    final = attempts[-1]
    root_cause = final.get("root_cause") if isinstance(final, Mapping) else final.root_cause
    root_value = root_cause.value if isinstance(root_cause, RootCause) else root_cause
    return normalize_terminal_failure_code(root_value)


def normalize_decision_execution_trace(
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    """返回 V2 内存投影；旧 trace 只读归一化且不修改输入。"""
    raw_attempts = trace.get("execution_attempts")
    if not isinstance(raw_attempts, (list, tuple)) or not raw_attempts:
        raise ValueError("decision trace requires execution attempts")
    failure_reason = trace.get("structured_failure_reason")
    if failure_reason is None:
        failure_reason = trace.get("terminal_failure_code")
    if all(isinstance(item, Mapping) for item in raw_attempts):
        translated = translate_serialized_decision_outcome(
            raw_attempts,
            structured_failure_reason=failure_reason,
        )
    elif all(not isinstance(item, Mapping) for item in raw_attempts):
        translated = translate_decision_outcome(
            tuple(raw_attempts),
            structured_failure_reason=failure_reason,
        )
    else:
        raise TypeError("execution attempts must share one schema")

    normalized = dict(trace)
    normalized.update(
        attempt_count=translated.attempt_count,
        retry_count=translated.retry_count,
        provider_fallback_count=translated.provider_fallback_count,
        decision_outcome=translated.outcome.value,
        generated_by=translated.generated_by.value,
        terminal_failure_code=translated.terminal_failure_code,
    )
    v2_fields = {
        "attempt_count",
        "provider_fallback_count",
        "generated_by",
        "terminal_failure_code",
    }
    if not v2_fields.issubset(trace):
        normalized["normalized_from_schema_version"] = "1"
    return normalized


def _route_kind(item: DecisionAttempt | Mapping[str, Any]) -> RouteKind:
    route_kind = item.get("route_kind") if isinstance(item, Mapping) else item.route_kind
    return route_kind if isinstance(route_kind, RouteKind) else RouteKind(route_kind)


def derive_generated_by(
    attempts: Sequence[DecisionAttempt | Mapping[str, Any]],
) -> DecisionGeneratedBy:
    """按最终 attempt 的最近非 RETRY 路线推导最终内容来源。"""
    if not attempts:
        raise ValueError("at least one attempt is required")
    source_by_route = {
        RouteKind.PRIMARY: DecisionGeneratedBy.MODEL,
        RouteKind.PROVIDER_FALLBACK: DecisionGeneratedBy.PROVIDER_FALLBACK,
        RouteKind.REPAIR: DecisionGeneratedBy.REPAIR,
        RouteKind.SAFE_FALLBACK: DecisionGeneratedBy.TERMINAL_FALLBACK,
    }
    for attempt in reversed(attempts):
        route = _route_kind(attempt)
        if route is not RouteKind.RETRY:
            return source_by_route[route]
    raise ValueError("retry attempt requires a non-retry source route")


def translate_serialized_decision_outcome(
    attempts: Sequence[Mapping[str, Any]],
    *,
    structured_failure_reason: str | None = None,
) -> TranslatedDecisionOutcome:
    """规范化脱敏 JSON 尝试后复用唯一 translator，拒绝未知枚举。"""
    normalized = tuple(_serialized_attempt(item) for item in attempts)
    return translate_decision_outcome(
        normalized,
        structured_failure_reason=structured_failure_reason,
    )


def _serialized_attempt(payload: Mapping[str, Any]) -> _SerializedDecisionAttempt:
    request_id = payload.get("opaque_request_id")
    if isinstance(request_id, Mapping):
        request_id = request_id.get("value")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("serialized attempt requires an opaque request id")
    return _SerializedDecisionAttempt(
        opaque_request_id=request_id,
        ordinal=_parse_serialized_ordinal(payload.get("ordinal")),
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


def _parse_serialized_ordinal(value: Any) -> int:
    """只接受正整数或无前导零的 ASCII 正整数字符串。"""
    if type(value) is int:
        ordinal = value
    elif (
        isinstance(value, str)
        and value
        and value.isascii()
        and value.isdecimal()
        and value[0] != "0"
    ):
        ordinal = int(value)
    else:
        raise ValueError("serialized attempt ordinal must be a canonical positive integer")
    if ordinal <= 0:
        raise ValueError("serialized attempt ordinal must be a canonical positive integer")
    return ordinal


__all__ = [
    "AttemptCounts",
    "DecisionGeneratedBy",
    "DecisionOutcome",
    "STABLE_TERMINAL_FAILURE_CODES",
    "DecisionAttempt",
    "TranslatedDecisionOutcome",
    "derive_generated_by",
    "derive_terminal_failure_code",
    "normalize_decision_execution_trace",
    "normalize_terminal_failure_code",
    "summarize_attempt_counts",
    "translate_decision_outcome",
    "translate_serialized_decision_outcome",
]
