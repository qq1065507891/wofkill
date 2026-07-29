# -*- coding: utf-8 -*-
"""
验证游戏修订版本、视图指纹和不可变读取引用契约。

作者: Project contributors
创建日期: 2026-07-29
"""

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.revisions import (
    ReadReference,
    RevisionContext,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def test_revision_context_is_strict_and_immutable() -> None:
    context = RevisionContext(
        base_revision=7,
        window_id="window-1",
        window_version=2,
        view_fingerprint=HASH_A,
    )
    assert context.base_revision == 7
    with pytest.raises(ValidationError):
        RevisionContext.model_validate({
            "base_revision": "7",
            "window_id": "window-1",
            "window_version": 2,
            "view_fingerprint": HASH_A,
        })
    with pytest.raises(ValidationError):
        context.base_revision = 8


def test_read_reference_rejects_invalid_hash_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ReadReference(record_id="record-1", revision=1, content_hash="short")
    with pytest.raises(ValidationError):
        ReadReference.model_validate({
            "record_id": "record-1",
            "revision": 1,
            "content_hash": HASH_B,
            "payload": "forbidden",
        })


def test_read_reference_accepts_revision_zero() -> None:
    reference = ReadReference(
        record_id="ruleset-snapshot",
        revision=0,
        content_hash=HASH_B,
    )
    assert reference.revision == 0
