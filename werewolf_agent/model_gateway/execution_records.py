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
    PROVIDER_METADATA = "provider_metadata"
    NORMALIZED_RESPONSE = "normalized_response"


@dataclass(frozen=True)
class AttemptExecutionRecord:
    """单次 provider 尝试的不可变、脱敏记录。"""

    request_id: str
    ordinal: int
    provider: str
    model: str
    route_kind: RouteKind
    root_cause: RootCause
    attempt_outcome: AttemptOutcome
    requested_reasoning_level: str
    normalized_reasoning_status: str
    reasoning_token_count: int
    evidence_kind: EvidenceKind

    def __post_init__(self) -> None:
        if not self.request_id or any(char.isspace() for char in self.request_id):
            raise ValueError("request_id must be an opaque non-empty token")
        if self.ordinal < 1:
            raise ValueError("ordinal must be >= 1")
        if self.reasoning_token_count < 0:
            raise ValueError("reasoning_token_count must be >= 0")
        if self.attempt_outcome is AttemptOutcome.FAILURE:
            if self.root_cause is RootCause.NONE:
                raise ValueError("failed attempt requires a root cause")
        elif self.root_cause is not RootCause.NONE:
            raise ValueError("successful attempt cannot have a root cause")


__all__ = [
    "AttemptExecutionRecord",
    "AttemptOutcome",
    "EvidenceKind",
    "RootCause",
    "RouteKind",
]
