"""Long-term reflection memory: post-game unstructured experience storage.

Design doc §10: unstructured reflections (e.g. "last time I trusted
emotional speech and got misaligned") go into long-term memory.
Vote chains, claims, and attack/defense logic stay structured in RelationGraph.

This module stores reflections with tag-based retrieval. Vector search
is a future extension — current implementation uses exact tag/role matching.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.memory.schemas import CrossGameQuery, ReflectionEntry


class ReflectionMemory:
    """Stores and retrieves post-game reflections across games."""

    def __init__(self) -> None:
        self._entries: dict[str, ReflectionEntry] = {}

    def store(self, entry: ReflectionEntry) -> None:
        self._entries[entry.entry_id] = entry

    def get(self, entry_id: str) -> ReflectionEntry | None:
        return self._entries.get(entry_id)

    def all_entries(self) -> list[ReflectionEntry]:
        return list(self._entries.values())

    def count(self) -> int:
        return len(self._entries)

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False

    def query(self, query: CrossGameQuery) -> list[ReflectionEntry]:
        """Retrieve reflections matching query criteria."""
        results = list(self._entries.values())

        if query.player_id:
            results = [e for e in results if e.player_id == query.player_id]

        if query.role:
            results = [e for e in results if e.role == query.role]

        if query.tags:
            results = [
                e for e in results
                if any(t in e.tags for t in query.tags)
            ]

        if query.situation:
            results = [
                e for e in results
                if query.situation.lower() in e.situation.lower()
            ]

        if query.faction_won is not None:
            results = [e for e in results if e.faction_won == query.faction_won]

        return results[:query.max_results]

    def by_player(self, player_id: str) -> list[ReflectionEntry]:
        return [e for e in self._entries.values() if e.player_id == player_id]

    def by_role(self, role: str) -> list[ReflectionEntry]:
        return [e for e in self._entries.values() if e.role == role]

    def by_game(self, game_id: str) -> list[ReflectionEntry]:
        return [e for e in self._entries.values() if e.game_id == game_id]

    def tag_index(self) -> dict[str, int]:
        """Return tag → count mapping for observability."""
        tag_counts: dict[str, int] = {}
        for entry in self._entries.values():
            for tag in entry.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts
