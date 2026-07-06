# -*- coding: utf-8 -*-
"""
提取技能 handler 需要的局势上下文查询 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.skills.skill_context import _alive_wolves
"""

from __future__ import annotations

from typing import Any


def _count_seer_claimants(ws: Any) -> int:
    """Count distinct players who publicly claimed seer."""
    if ws is None:
        return 0
    claimants: set[str] = set()
    for f in ws.facts_of_type("claimed_role"):
        if f.value == "seer" and f.source_player:
            claimants.add(f.source_player)
    return len(claimants)

def _get_seer_claimants(ws: Any) -> list[str]:
    """Return list of players who publicly claimed seer."""
    if ws is None:
        return []
    claimants: set[str] = set()
    for f in ws.facts_of_type("claimed_role"):
        if f.value == "seer" and f.source_player:
            claimants.add(f.source_player)
    return sorted(claimants)

def _alive_wolves(gs: Any) -> list[str]:
    """Return alive wolf teammates."""
    if gs is None:
        return []
    return [
        pid for pid, p in gs.players.items()
        if p.alive and p.role == "werewolf"
    ]

def _alive_non_wolves(gs: Any) -> list[str]:
    """Return alive non-wolf players."""
    if gs is None:
        return []
    return [
        pid for pid, p in gs.players.items()
        if p.alive and p.role != "werewolf"
    ]

def _vote_targets_for_player(ws: Any, player_id: str) -> list[dict[str, Any]]:
    """Get vote facts targeting a specific player."""
    if ws is None:
        return []
    return [
        {"source": f.source_player, "day": f.day, "value": f.value}
        for f in ws.facts_of_type("vote")
        if f.target_player == player_id
    ]

def _seer_checks_on_target(ws: Any, target_id: str) -> list[dict[str, Any]]:
    """Get seer check claims targeting a specific player."""
    if ws is None:
        return []
    results = []
    for f in ws.facts_of_type("seer_check_claim"):
        if f.target_player == target_id:
            results.append({"source": f.source_player, "value": f.value, "day": f.day})
    return results

def _alerts_for_player(alerts: list[Any], player_id: str) -> list[Any]:
    """Filter contradiction alerts that mention the player."""
    if not player_id:
        return []
    return [
        a for a in alerts
        if player_id == a.player_id or ("," in a.player_id and player_id in a.player_id.split(","))
    ]

def _belief_top_suspects(bs: Any, count: int = 3) -> list[tuple[str, str, float]]:
    """Return top suspects from belief state (wolf_lean, lowest trust)."""
    if bs is None:
        return []
    suspects: list[tuple[str, str, float]] = []
    for pid, belief in bs.beliefs.items():
        if belief.faction_lean == "wolf_lean" or belief.trust < 0.35:
            suspects.append((pid, belief.faction_lean, belief.trust))
    suspects.sort(key=lambda x: x[2])
    return suspects[:count]

def _wolf_teammates_exposed(ws: Any, wolf_ids: list[str]) -> list[dict[str, Any]]:
    """Check which wolf teammates have been publicly seer-checked as wolf."""
    if ws is None:
        return []
    exposed = []
    for wid in wolf_ids:
        checks = _seer_checks_on_target(ws, wid)
        for c in checks:
            if "wolf" in c.get("value", "").lower() or "狼" in c.get("value", ""):
                exposed.append({"teammate": wid, "checked_by": c["source"]})
    return exposed
