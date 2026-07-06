# -*- coding: utf-8 -*-
"""
从事件日志计算游戏节奏指标。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.evaluation.pace_metrics import compute_pace_metrics
    >>> compute_pace_metrics([], finish_night=None)["pace_target_met"]
    False
"""

from __future__ import annotations

from typing import Any


def compute_pace_metrics(
    events: list[dict[str, Any]],
    *,
    deaths: list[dict[str, Any]] | None = None,
    finish_night: int | None = None,
) -> dict[str, Any]:
    """从事件日志计算游戏节奏指标。"""
    vote_events = [
        e for e in events if e.get("type") == "vote_resolved"
    ]

    total_vote_days = len(vote_events)
    exile_days = sum(
        1 for e in vote_events
        if e.get("payload", {}).get("exiled") is not None
    )
    second_tie_count = sum(
        1 for e in vote_events
        if e.get("payload", {}).get("reason") == "second_tie_no_exile"
    )

    day_exile_rate = exile_days / total_vote_days if total_vote_days > 0 else 0.0

    max_streak = 0
    current_streak = 0
    for e in vote_events:
        if e.get("payload", {}).get("exiled") is None:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    stale_count = 0
    seen_votes: list[dict] = []
    for e in events:
        if e.get("type") == "vote_resolved":
            votes_snapshot = e.get("payload", {}).get("votes", {})
            if votes_snapshot:
                for prev in seen_votes:
                    if votes_snapshot == prev:
                        stale_count += 1
                        break
                seen_votes.append(votes_snapshot)

    pace_target_met = (
        (finish_night is not None and finish_night <= 8)
        and max_streak <= 1
        and stale_count == 0
        and (total_vote_days < 3 or day_exile_rate >= 0.5)
    )

    return {
        "day_exile_rate": round(day_exile_rate, 3),
        "max_consecutive_no_exile_days": max_streak,
        "second_tie_count": second_tie_count,
        "stale_vote_reuse_count": stale_count,
        "finish_night_number": finish_night,
        "pace_target_met": pace_target_met,
    }


__all__ = [
    "compute_pace_metrics",
]
