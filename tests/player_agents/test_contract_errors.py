# -*- coding: utf-8 -*-
"""
验证提案校验错误使用稳定代码、JSON Pointer 路径且不携带隐藏值。

作者: Project contributors
创建日期: 2026-07-29
"""

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.contracts import (
    ProposalFailure as ExportedProposalFailure,
)
from werewolf_agent.player_agents.contracts import (
    ValidationErrorCode as ExportedValidationErrorCode,
)
from werewolf_agent.player_agents.contracts.errors import (
    ProposalFailure,
    ValidationErrorCode,
)


def test_failure_serializes_stable_code_and_field_path() -> None:
    failure = ProposalFailure.for_code(
        code=ValidationErrorCode.TARGET_NOT_LEGAL,
        field_path="/body/moves/0/target_id",
        repairable=False,
    )
    assert failure.model_dump(mode="json")["code"] == "target_not_legal"
    assert failure.message == "target is not legal for this window"


def test_failure_rejects_non_json_pointer_and_extra_context() -> None:
    with pytest.raises(ValidationError):
        ProposalFailure(
            code=ValidationErrorCode.INVISIBLE_REFERENCE,
            field_path="body.secret",
            message="reference is not visible",
            repairable=False,
        )
    with pytest.raises(ValidationError) as exc_info:
        ProposalFailure.model_validate({
            "code": ValidationErrorCode.INVISIBLE_REFERENCE,
            "field_path": "/body/ref",
            "message": "reference is not visible",
            "repairable": False,
            "hidden_value": "seer:p03",
        })
    assert any(
        error["loc"] == ("hidden_value",)
        and error["type"] == "extra_forbidden"
        for error in exc_info.value.errors()
    )


def test_failure_rejects_unsafe_message() -> None:
    with pytest.raises(ValidationError, match="safe message catalog"):
        ProposalFailure(
            code=ValidationErrorCode.INVISIBLE_REFERENCE,
            field_path="/body/ref",
            message="hidden role is seer:p03",
            repairable=False,
        )


def test_stale_failure_is_not_repairable() -> None:
    with pytest.raises(ValidationError, match="not repairable"):
        ProposalFailure.for_code(
            code=ValidationErrorCode.STALE_READ_SET,
            field_path="/read_set",
            repairable=True,
        )


def test_root_semantic_failure_is_not_repairable() -> None:
    with pytest.raises(ValidationError, match="field-local"):
        ProposalFailure.for_code(
            code=ValidationErrorCode.SEMANTIC_MISMATCH,
            field_path="",
            repairable=True,
        )


def test_failure_allows_only_declared_repair_scopes() -> None:
    schema_failure = ProposalFailure.for_code(
        code=ValidationErrorCode.SCHEMA_INVALID,
        field_path="",
        repairable=True,
    )
    semantic_failure = ProposalFailure.for_code(
        code=ValidationErrorCode.SEMANTIC_MISMATCH,
        field_path="/body/moves/0/target_id",
        repairable=True,
    )
    assert schema_failure.repairable is True
    assert semantic_failure.repairable is True


def test_failure_public_export_is_frozen_and_round_trips_json() -> None:
    failure = ExportedProposalFailure.for_code(
        code=ExportedValidationErrorCode.INVISIBLE_REFERENCE,
        field_path="/body/ref",
        repairable=False,
    )
    assert ExportedValidationErrorCode is ValidationErrorCode
    with pytest.raises(ValidationError):
        failure.message = "replacement"
    assert ExportedProposalFailure.model_validate_json(
        failure.model_dump_json()
    ) == failure
