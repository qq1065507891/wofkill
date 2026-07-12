# -*- coding: utf-8 -*-
"""
定义隐私安全、强类型的模型逐次尝试执行记录。

作者: Project contributors
创建日期: 2026-07-13

使用示例:
    >>> record.ordinal
    1
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


@dataclass(frozen=True)
class OpaqueRequestId:
    """仅允许内部 run-scope 格式的脱敏请求标识。"""

    value: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"run_[a-z0-9]{4,32}_[a-f0-9]{8,32}", self.value):
            raise ValueError("opaque_request_id must use the internal run-scope format")


class RootCause(str, Enum):
    """一次失败尝试的权威根因。"""

    NONE = "none"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    INVALID_OUTPUT = "invalid_output"
    POLICY_REJECTION = "policy_rejection"


class AttemptOutcome(str, Enum):
    """单次尝试事实，不包含最终决策语义。"""

    SUCCESS = "attempt_success"
    FAILURE = "attempt_failure"


class RouteKind(str, Enum):
    """本次尝试在调用路线中的位置。"""

    PRIMARY = "primary"
    RETRY = "retry"
    PROVIDER_FALLBACK = "provider_fallback"
    REPAIR = "repair"
    SAFE_FALLBACK = "safe_fallback"


class EvidenceKind(str, Enum):
    """推理状态证据的来源类别，不保存原始私密内容。"""

    NONE = "none"
    AUTHORITATIVE_PROVIDER_EXECUTION = "authoritative_provider_execution"
    NORMALIZED_RESPONSE = "normalized_response"


class ReasoningLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReasoningStatus(str, Enum):
    CONFIRMED = "confirmed"
    NOT_REQUESTED = "not_requested"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class AttemptExecutionRecord:
    """单次 provider 尝试的不可变、脱敏记录。"""

    opaque_request_id: OpaqueRequestId
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

    def __post_init__(self) -> None:
        if not isinstance(self.opaque_request_id, OpaqueRequestId):
            raise ValueError("opaque_request_id must be created by the internal boundary")
        if self.ordinal < 1:
            raise ValueError("ordinal must be >= 1")
        if self.reasoning_token_count < 0:
            raise ValueError("reasoning_token_count must be >= 0")
        if self.attempt_outcome is AttemptOutcome.FAILURE:
            if self.root_cause is RootCause.NONE:
                raise ValueError("failed attempt requires a root cause")
        elif self.root_cause is not RootCause.NONE:
            raise ValueError("successful attempt cannot have a root cause")
        if self.normalized_reasoning_status is ReasoningStatus.CONFIRMED:
            if self.reasoning_token_count == 0 and self.evidence_kind is not EvidenceKind.AUTHORITATIVE_PROVIDER_EXECUTION:
                raise ValueError("confirmed reasoning requires tokens or authoritative execution evidence")
        if self.normalized_reasoning_status is ReasoningStatus.NOT_REQUESTED:
            if self.reasoning_token_count or self.evidence_kind is not EvidenceKind.NONE:
                raise ValueError("not_requested reasoning cannot carry provider evidence")


__all__ = [
    "AttemptExecutionRecord",
    "AttemptOutcome",
    "EvidenceKind",
    "OpaqueRequestId",
    "ReasoningLevel",
    "ReasoningStatus",
    "RootCause",
    "RouteKind",
]
