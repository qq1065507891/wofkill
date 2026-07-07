# -*- coding: utf-8 -*-
"""
计算阵营、玩家和角色胜率等基础 outcome 指标。

作者: Project contributors
创建日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.evaluation.metric_outcomes import compute_faction_metrics
    >>> compute_faction_metrics(aggregator, snapshot)
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.evaluation.schemas import (
    FactionMetrics,
    MetricsSnapshot,
    PlayerMetrics,
    RoleMetrics,
)


def compute_faction_metrics(aggregator: Any, snap: MetricsSnapshot) -> None:
    """计算整体好人/狼人阵营胜率。"""
    total = len(aggregator._results)
    good_wins = sum(1 for r in aggregator._results if r.winning_faction == "good")
    wolf_wins = sum(1 for r in aggregator._results if r.winning_faction == "werewolf")
    snap.faction_metrics = FactionMetrics(
        good_win_rate=good_wins / total if total else 0.0,
        werewolf_win_rate=wolf_wins / total if total else 0.0,
        good_wins=good_wins,
        werewolf_wins=wolf_wins,
        total_games=total,
    )


def compute_player_metrics(aggregator: Any, snap: MetricsSnapshot) -> None:
    """按玩家聚合总胜率和分角色胜率。"""
    player_stats: dict[str, dict[str, Any]] = {}

    for result in aggregator._results:
        winner = result.winning_faction
        for pid, faction in result.player_factions.items():
            if pid not in player_stats:
                player_stats[pid] = {"games": 0, "wins": 0, "role_stats": {}}
            stats = player_stats[pid]
            stats["games"] += 1
            if faction == winner:
                stats["wins"] += 1
            role = result.player_roles.get(pid, "unknown")
            rs = stats["role_stats"]
            if role not in rs:
                rs[role] = {"games": 0, "wins": 0}
            rs[role]["games"] += 1
            if faction == winner:
                rs[role]["wins"] += 1

    for pid, stats in player_stats.items():
        games = stats["games"]
        wins = stats["wins"]
        pm = PlayerMetrics(
            player_id=pid,
            win_rate=wins / games if games else 0.0,
            games=games,
            wins=wins,
        )
        for role, rs in stats["role_stats"].items():
            pm.role_metrics[role] = RoleMetrics(
                role=role,
                win_rate=rs["wins"] / rs["games"] if rs["games"] else 0.0,
                games=rs["games"],
                wins=rs["wins"],
            )
        snap.player_metrics[pid] = pm


def compute_role_metrics(aggregator: Any, snap: MetricsSnapshot) -> None:
    """按角色聚合胜率。"""
    role_stats: dict[str, dict[str, int]] = {}

    for result in aggregator._results:
        winner = result.winning_faction
        for pid, role in result.player_roles.items():
            if role not in role_stats:
                role_stats[role] = {"games": 0, "wins": 0}
            role_stats[role]["games"] += 1
            faction = result.player_factions.get(pid, "")
            if faction == winner:
                role_stats[role]["wins"] += 1

    for role, stats in role_stats.items():
        snap.role_metrics[role] = RoleMetrics(
            role=role,
            win_rate=stats["wins"] / stats["games"] if stats["games"] else 0.0,
            games=stats["games"],
            wins=stats["wins"],
        )


__all__ = [
    "compute_faction_metrics",
    "compute_player_metrics",
    "compute_role_metrics",
]
