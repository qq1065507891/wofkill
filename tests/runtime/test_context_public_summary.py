# -*- coding: utf-8 -*-
"""
验证玩家上下文公开摘要和近期记录构建逻辑。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-25

使用示例:
    >>> from werewolf_agent.runtime.context_public_summary import build_public_summary
    >>> build_public_summary(game_state)
"""

from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.context_public_summary import (
    build_public_summary,
    build_recent_transcript,
)


def _legacy_vote_payload() -> dict[str, object]:
    """构造仅含内部单位别名的 V1 投票载荷。"""
    return {
        "day_number": 1,
        "exiled": "p02",
        "weighted_tally": {"p02": 3, "p03": 2},
        "vote_weights": {"p01": 3, "p03": 2},
    }


def _v2_vote_payload() -> dict[str, object]:
    """构造同时携带别名、单位和展示值的 V2 投票载荷。"""
    return {
        **_legacy_vote_payload(),
        "vote_weight_format_version": 2,
        "base_vote_weight": 2,
        "weighted_tally_units": {"p02": 3, "p03": 2},
        "vote_weight_units": {"p01": 3, "p03": 2},
        "weighted_tally_display": {"p02": 1.5, "p03": 1},
        "vote_weights_display": {"p01": 1.5, "p03": 1},
    }


def test_build_recent_transcript_keeps_speeches_and_vote_records_in_order():
    gs = GameState(events=[
        GameEvent("speech", {"speaker": "p01", "text": "我怀疑p02"}),
        GameEvent("vote_resolved", {
            "day_number": 1,
            "exiled": "p02",
            "votes": [
                {"voter": "p01", "target": "p02"},
                {"voter": "p03", "target": None},
            ],
        }),
    ])

    assert build_recent_transcript(gs) == [
        {"speaker": "p01", "text": "我怀疑p02", "type": "speech"},
        {
            "type": "vote_record",
            "day": 1,
            "result": "p02",
            "votes": {"p01": "p02", "p03": None},
        },
    ]


def test_public_context_excludes_v2_private_event_without_payload_visibility():
    gs = GameState(events=[GameEvent(
        type="speech",
        payload={"speaker": "p01", "text": "私密发言"},
        visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
        schema_version="2",
    )])

    assert build_recent_transcript(gs) == []
    assert "私密发言" not in build_public_summary(gs)


def test_build_public_summary_prioritizes_vote_details_and_prepends_timeline_note():
    gs = GameState(events=[
        GameEvent("day_announce", {"day": 1}),
        GameEvent("vote_resolved", {
            "day_number": 1,
            "exiled": "p02",
            **_v2_vote_payload(),
            "votes": [
                {"voter": "p01", "target": "p02"},
                {"voter": "p03", "target": None},
            ],
        }),
    ])

    summary = build_public_summary(gs)

    assert summary.startswith("时间顺序为 N1 首夜")
    assert "[放逐] D1 p02被放逐 (p02=1.5票、p03=1票)" in summary
    assert "p02=3票" not in summary
    assert "[投票] D1: p01→p02，p03→弃票" in summary


def test_public_summary_does_not_expose_unknown_v1_internal_vote_units() -> None:
    summary = build_public_summary(
        GameState(events=[GameEvent("vote_resolved", _legacy_vote_payload())])
    )

    assert "[放逐] D1 p02被放逐" in summary
    assert "p02=3票" not in summary


def test_public_summary_ignores_conflicting_v2_vote_aliases() -> None:
    payload = _v2_vote_payload()
    payload["weighted_tally"] = {"p02": 6, "p03": 2}

    summary = build_public_summary(
        GameState(events=[GameEvent("vote_resolved", payload)])
    )

    assert "[放逐] D1 p02被放逐" in summary
    assert "p02=3票" not in summary
    assert "p02=1.5票" not in summary
