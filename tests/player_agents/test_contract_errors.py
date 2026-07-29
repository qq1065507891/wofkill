# -*- coding: utf-8 -*-
"""
验证提案校验错误使用稳定代码、JSON Pointer 路径且不携带隐藏值。

作者: Project contributors
创建日期: 2026-07-29
"""

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.errors import (
    ProposalFailure,
    ValidationErrorCode,
)


def test_failure_serializes_stable_code_and_field_path() -> None:
    failure = ProposalFailure(
        code=ValidationErrorCode.TARGET_NOT_LEGAL,
        field_path="/body/moves/0/target_id",
        message="target is not legal for this window",
        repairable=False,
    )
    assert failure.model_dump(mode="json")["code"] == "target_not_legal"


def test_failure_rejects_non_json_pointer_and_extra_context() -> None:
    with pytest.raises(ValidationError):
        ProposalFailure(
            code=ValidationErrorCode.INVISIBLE_REFERENCE,
            field_path="body.secret",
            message="reference is not visible",
            repairable=False,
        )
    with pytest.raises(ValidationError):
        ProposalFailure.model_validate({
            "code": "invisible_reference",
            "field_path": "/body/ref",
            "message": "reference is not visible",
            "repairable": False,
            "hidden_value": "seer:p03",
        })
