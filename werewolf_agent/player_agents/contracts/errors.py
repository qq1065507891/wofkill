# -*- coding: utf-8 -*-
"""
定义玩家终端提案校验的稳定错误代码和安全失败载荷。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import StringConstraints, model_validator

from werewolf_agent.player_agents.contracts._base import StrictFrozenModel

JsonPointer = Annotated[str, StringConstraints(pattern=r"^(|/(?:[^~/]|~0|~1)*)+$")]


class ValidationErrorCode(StrEnum):
    SCHEMA_INVALID = "schema_invalid"
    BOUND_CONTEXT_MISMATCH = "bound_context_mismatch"
    UNKNOWN_SCHEMA_VERSION = "unknown_schema_version"
    UNKNOWN_CAPABILITY = "unknown_capability"
    WRONG_ACTION_WINDOW = "wrong_action_window"
    STALE_READ_SET = "stale_read_set"
    TARGET_NOT_LEGAL = "target_not_legal"
    INVISIBLE_REFERENCE = "invisible_reference"
    GRANT_INACTIVE = "grant_inactive"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    RULE_ILLEGAL = "rule_illegal"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    SECURITY_VIOLATION = "security_violation"


_SAFE_MESSAGES: dict[ValidationErrorCode, str] = {
    ValidationErrorCode.SCHEMA_INVALID: "proposal does not match the required schema",
    ValidationErrorCode.BOUND_CONTEXT_MISMATCH: "proposal context does not match the bound turn",
    ValidationErrorCode.UNKNOWN_SCHEMA_VERSION: "proposal schema version is not supported",
    ValidationErrorCode.UNKNOWN_CAPABILITY: "proposal capability is not supported",
    ValidationErrorCode.WRONG_ACTION_WINDOW: "proposal is not legal for the current window",
    ValidationErrorCode.STALE_READ_SET: "proposal read set is stale",
    ValidationErrorCode.TARGET_NOT_LEGAL: "target is not legal for this window",
    ValidationErrorCode.INVISIBLE_REFERENCE: "reference is not visible",
    ValidationErrorCode.GRANT_INACTIVE: "disclosure grant is not active",
    ValidationErrorCode.SEMANTIC_MISMATCH: "proposal field does not match its semantic constraints",
    ValidationErrorCode.RULE_ILLEGAL: "proposal is not legal under the current rules",
    ValidationErrorCode.IDEMPOTENCY_CONFLICT: "idempotency key conflicts with an existing proposal",
    ValidationErrorCode.SECURITY_VIOLATION: "proposal violates a security constraint",
}


class ProposalFailure(StrictFrozenModel):
    code: ValidationErrorCode
    field_path: JsonPointer
    message: Annotated[str, StringConstraints(min_length=1, max_length=240)]
    repairable: bool

    @classmethod
    def for_code(
        cls,
        *,
        code: ValidationErrorCode,
        field_path: str,
        repairable: bool,
    ) -> ProposalFailure:
        """使用封闭的安全消息目录构造失败载荷。"""
        return cls(
            code=code,
            field_path=field_path,
            message=_SAFE_MESSAGES[code],
            repairable=repairable,
        )

    @model_validator(mode="after")
    def _safe_repair_contract(self) -> ProposalFailure:
        if self.message != _SAFE_MESSAGES[self.code]:
            raise ValueError("message must come from the safe message catalog")
        if self.repairable and self.code not in {
            ValidationErrorCode.SCHEMA_INVALID,
            ValidationErrorCode.SEMANTIC_MISMATCH,
        }:
            raise ValueError(f"{self.code.value} is not repairable")
        if (
            self.repairable
            and self.code is ValidationErrorCode.SEMANTIC_MISMATCH
            and self.field_path == ""
        ):
            raise ValueError("repairable semantic_mismatch must be field-local")
        return self
