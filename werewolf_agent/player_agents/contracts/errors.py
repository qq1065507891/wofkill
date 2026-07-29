# -*- coding: utf-8 -*-
"""
定义玩家终端提案校验的稳定错误代码和安全失败载荷。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import StringConstraints

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


class ProposalFailure(StrictFrozenModel):
    code: ValidationErrorCode
    field_path: JsonPointer
    message: Annotated[str, StringConstraints(min_length=1, max_length=240)]
    repairable: bool
