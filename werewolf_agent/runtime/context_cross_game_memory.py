# -*- coding: utf-8 -*-
"""
组装 AgentContext 所需的跨局画像、反思卡片和认知矩阵提示。

作者: Project contributors
创建日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.runtime.context_cross_game_memory import build_cross_game_memory_hints
    >>> build_cross_game_memory_hints(restored_memory, "p01", "seer")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from werewolf_agent.memory.store import MemoryStore
from werewolf_agent.runtime.context_memory_hints import (
    REFLECTION_CARD_BUDGET,
    _cognition_matrix_hint,
    _profile_memory_hint,
    _reflection_memory_hints,
)


@dataclass(frozen=True)
class CrossGameMemoryHints:
    """跨局记忆注入结果，字段直接对应 AgentContext 的提示字段。"""

    profile_memory_hint: dict[str, Any] = field(default_factory=dict)
    reflection_memory_hints: list[dict[str, Any]] = field(default_factory=list)
    cognition_matrix_hint: dict[str, Any] = field(default_factory=dict)
    error_pattern_hint: dict[str, Any] = field(default_factory=dict)


def _role_stats_for_refs(refs: list[Any]) -> dict[str, dict[str, int]]:
    """按角色聚合反思样本数量和胜局数。"""
    role_stats: dict[str, dict[str, int]] = {}
    for ref in refs:
        role = getattr(ref, "role", None) or "?"
        stats = role_stats.setdefault(role, {"count": 0, "wins": 0})
        stats["count"] += 1
        if getattr(ref, "faction_won", False):
            stats["wins"] += 1
    return role_stats


def _query_live_reflections(
    reflection_memory: Any,
    *,
    player_id: str,
    current_role: str,
) -> list[Any]:
    """按实时 prompt 预算查询 V2 反思卡片。"""
    if reflection_memory is None or not hasattr(reflection_memory, "query_live"):
        return []
    from werewolf_agent.memory.schemas import CrossGameQuery

    return reflection_memory.query_live(
        CrossGameQuery(
            player_id=player_id,
            role=current_role,
            max_results=REFLECTION_CARD_BUDGET,
        )
    )


def build_cross_game_memory_hints(
    restored_memory: Any,
    *,
    player_id: str,
    current_role: str,
) -> CrossGameMemoryHints:
    """从持久化记忆中提取当前玩家的跨局提示。"""
    if restored_memory is None:
        return CrossGameMemoryHints()

    profile = restored_memory.get_profile(player_id)
    current_faction = MemoryStore._player_faction(
        current_role,
        master_faction=None,
    )

    reflection_memory = getattr(restored_memory, "reflections", None)
    v2_refs = _query_live_reflections(
        reflection_memory,
        player_id=player_id,
        current_role=current_role,
    )

    reflection_memory_hints: list[dict[str, Any]] = []
    error_pattern_hint: dict[str, Any] = {}
    if v2_refs:
        reflection_memory_hints = _reflection_memory_hints(
            v2_refs,
            current_role,
            current_faction,
        )
        live_error_pattern = getattr(reflection_memory, "live_error_pattern", None)
        if callable(live_error_pattern):
            error_pattern_hint = live_error_pattern(player_id, current_role)

    profile_memory_hint: dict[str, Any] = {}
    if profile is not None and profile.games_played > 0:
        refs_for_profile: list[Any] = []
        reflections_by_player = getattr(restored_memory, "reflections_by_player", None)
        if callable(reflections_by_player):
            refs_for_profile = reflections_by_player(player_id)
        elif v2_refs:
            refs_for_profile = v2_refs
        profile_memory_hint = _profile_memory_hint(
            profile,
            _role_stats_for_refs(refs_for_profile),
            current_role,
        )

    return CrossGameMemoryHints(
        profile_memory_hint=profile_memory_hint,
        reflection_memory_hints=reflection_memory_hints,
        cognition_matrix_hint=_cognition_matrix_hint(restored_memory, player_id),
        error_pattern_hint=error_pattern_hint,
    )


__all__ = [
    "CrossGameMemoryHints",
    "build_cross_game_memory_hints",
]
