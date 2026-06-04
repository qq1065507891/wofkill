"""Tests for werewolf_agent.runtime.context reflection memory hints sorting."""

from __future__ import annotations

from typing import Any

from werewolf_agent.memory.schemas import ReflectionEntry
from werewolf_agent.runtime.context import _reflection_memory_hints


def _make_reflection(
    *, game_id: str, role: str, player_id: str = "p01", text: str = "t", faction_won: bool = False
) -> ReflectionEntry:
    return ReflectionEntry(
        entry_id=f"reflection_{game_id}_{player_id}",
        game_id=game_id,
        player_id=player_id,
        role=role,
        faction_won=faction_won,
        text=text,
        situation=text,
    )


def test_reflection_hints_orders_newer_game_id_first_within_same_priority() -> None:
    """Within same priority bucket, newer game_id should sort first."""
    refs = [
        _make_reflection(game_id="2024-12-01", role="seer", text="old"),
        _make_reflection(game_id="2024-12-02", role="seer", text="middle"),
        _make_reflection(game_id="2024-12-03", role="seer", text="new"),
    ]

    hints = _reflection_memory_hints(refs, current_role="seer", current_faction="good")

    assert [h["text"] for h in hints] == ["new", "middle", "old"]


def test_reflection_hints_tie_broken_by_game_id_descending() -> None:
    """Same role + faction priority; ties broken by game_id descending."""
    refs = [
        _make_reflection(game_id="2024-11-15", role="seer", text="older"),
        _make_reflection(game_id="2024-12-20", role="seer", text="newer"),
    ]

    hints = _reflection_memory_hints(refs, current_role="seer", current_faction="good")

    assert [h["text"] for h in hints] == ["newer", "older"]


def test_reflection_hints_same_game_id_orders_by_entry_id_stable() -> None:
    """Within same game_id, sort is stable via entry_id (ascending)."""
    refs = [
        _make_reflection(
            game_id="2024-12-01", role="seer", player_id="p02", text="p02"
        ),
        _make_reflection(
            game_id="2024-12-01", role="seer", player_id="p01", text="p01"
        ),
    ]

    hints = _reflection_memory_hints(refs, current_role="seer", current_faction="good")

    # entry_id for p01 < entry_id for p02, so p01 first
    assert [h["text"] for h in hints] == ["p01", "p02"]


def test_reflection_hints_higher_priority_wins_over_newer_game() -> None:
    """Same role > same faction > other; higher priority dominates game_id recency."""
    refs = [
        # priority 0 (other), but very new game
        _make_reflection(game_id="2024-12-30", role="hunter", text="new-other"),
        # priority 2 (same role), but older game
        _make_reflection(game_id="2024-12-01", role="seer", text="old-same"),
    ]

    hints = _reflection_memory_hints(refs, current_role="seer", current_faction="good")

    # Priority 2 (same role) must beat priority 0 (other), even when other is newer
    assert [h["text"] for h in hints] == ["old-same", "new-other"]
