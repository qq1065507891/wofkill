# -*- coding: utf-8 -*-
"""
定义模型调用逐次尝试的强类型执行记录。

作者: Project contributors
创建日期: 2026-07-13

使用示例:
    >>> AttemptExecutionRecord(1, "openai", "gpt", AttemptOutcome.SUCCESS)
    AttemptExecutionRecord(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RootCause(str, Enum):
    """一次尝试失败的权威根因。"""

    NONE = "none"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    INVALID_OUTPUT = "invalid_output"
    POLICY_REJECTION = "policy_rejection"


class AttemptOutcome(str, Enum):
    """一次模型尝试的互斥结果。"""

    SUCCESS = "success"
    RETRYABLE_FAILURE = "retryable_failure"
    REPAIRED_SUCCESS = "repaired_success"
    PROVIDER_FALLBACK_SUCCESS = "provider_fallback_success"
    TERMINAL_FALLBACK = "terminal_fallback"

    @property
    def is_success(self) -> bool:
        """返回该结果是否已产生可用决策。"""
        return self is not AttemptOutcome.RETRYABLE_FAILURE


@dataclass(frozen=True)
class AttemptExecutionRecord:
    """单次 provider 尝试的不可变记录。"""

    attempt_number: int
    provider: str
    model: str
    outcome: AttemptOutcome
    root_cause: RootCause = RootCause.NONE

    def __post_init__(self) -> None:
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be >= 1")
        if self.outcome is AttemptOutcome.RETRYABLE_FAILURE:
            if self.root_cause is RootCause.NONE:
                raise ValueError("failed attempt requires a root cause")
        elif self.root_cause is not RootCause.NONE:
            raise ValueError("successful or fallback attempt cannot have a root cause")


__all__ = ["AttemptExecutionRecord", "AttemptOutcome", "RootCause"]
