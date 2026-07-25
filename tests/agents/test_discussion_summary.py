# -*- coding: utf-8 -*-
"""
验证讨论摘要 V2 Schema、旧 checkpoint 迁移和文本兼容投影。

作者: Project contributors
创建日期: 2026-07-25

使用示例:
    >>> python -m pytest tests/agents/test_discussion_summary.py -q
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from werewolf_agent.agents.discussion_summary import (
    DiscussionSummary,
    discussion_summary_for_player,
    discussion_summary_text,
)


def _v2_payload(summary: str = "我怀疑p03") -> dict[str, object]:
    return {
        "summary": summary,
        "suspected_players": ["p03"],
        "trusted_players": ["p02"],
        "vote_target": "p03",
        "evidence_refs": ["event-7"],
    }


def test_discussion_summary_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DiscussionSummary.model_validate({
            **_v2_payload(),
            "private_reason": "我是狼人，所以要推动p03。",
        })


def test_legacy_summary_string_upgrades_to_v2() -> None:
    state = {
        "discussion_positions": {
            "p01": "我怀疑p03",
            "p02": "我信任p04",
        },
    }

    summary = discussion_summary_for_player(state, "p01")

    assert summary == DiscussionSummary(summary="我怀疑p03")
    assert state["discussion_positions_version"] == 2
    assert state["discussion_positions"] == {
        "p01": DiscussionSummary(summary="我怀疑p03").model_dump(),
        "p02": DiscussionSummary(summary="我信任p04").model_dump(),
    }


def test_unversioned_v2_mapping_upgrades_in_memory() -> None:
    state = {
        "discussion_positions": {
            "p01": _v2_payload(),
            "p02": _v2_payload("我信任p03"),
        },
    }

    summary = discussion_summary_for_player(state, "p02")

    assert summary == DiscussionSummary.model_validate(_v2_payload("我信任p03"))
    assert state["discussion_positions_version"] == 2
    assert state["discussion_positions"]["p02"] == _v2_payload("我信任p03")


def test_explicit_v2_payload_is_read_without_coercion() -> None:
    state = {
        "discussion_positions_version": 2,
        "discussion_positions": {"p01": _v2_payload()},
    }

    assert discussion_summary_for_player(
        state,
        "p01",
    ) == DiscussionSummary.model_validate(_v2_payload())


def test_v2_version_schema_conflict_fails_closed() -> None:
    state = {
        "discussion_positions_version": 2,
        "discussion_positions": {"p01": "legacy string"},
    }

    assert discussion_summary_for_player(state, "p01") is None
    assert state["discussion_positions"]["p01"] == "legacy string"


def test_unknown_explicit_version_fails_closed() -> None:
    state = {
        "discussion_positions_version": 3,
        "discussion_positions": {"p01": _v2_payload()},
    }

    assert discussion_summary_for_player(state, "p01") is None


def test_mixed_unversioned_entries_validate_per_player_without_upgrade() -> None:
    state = {
        "discussion_positions": {
            "p01": _v2_payload(),
            "p02": {"summary": "字段冲突", "unexpected": True},
        },
    }

    assert discussion_summary_for_player(
        state,
        "p01",
    ) == DiscussionSummary.model_validate(_v2_payload())
    assert discussion_summary_for_player(state, "p02") is None
    assert "discussion_positions_version" not in state


def test_missing_player_returns_none() -> None:
    assert discussion_summary_for_player(
        {"discussion_positions": {}},
        "p09",
    ) is None


def test_text_projection_is_deterministic_and_excludes_private_fields() -> None:
    summary = DiscussionSummary.model_validate(_v2_payload())

    first = discussion_summary_text(summary)
    second = discussion_summary_text(summary)

    assert first == second
    assert first == (
        "我怀疑p03\n"
        "怀疑玩家: p03\n"
        "信任玩家: p02\n"
        "投票目标: p03\n"
        "证据引用: event-7"
    )
    assert "private" not in first.lower()
    assert "我是狼人" not in first
