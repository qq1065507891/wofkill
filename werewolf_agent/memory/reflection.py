"""Long-term reflection memory: post-game unstructured experience storage.

Design doc §10: unstructured reflections (e.g. "last time I trusted
emotional speech and got misaligned") go into long-term memory.
Vote chains, claims, and attack/defense logic stay structured in RelationGraph.

This module stores reflections with tag-based retrieval. Vector search
is a future extension — current implementation uses exact tag/role matching.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from werewolf_agent.memory.schemas import CrossGameQuery, ReflectionEntry


class ReflectionMemory:
    """Stores and retrieves post-game reflections across games.

    When a ``repo`` (PostgresGameRepository) is provided, reflections are
    persisted to the ``reflections`` table.  Otherwise entries are kept
    in memory only (suitable for tests and single-run scenarios).
    """

    def __init__(self, repo: Any | None = None) -> None:
        self._entries: dict[str, ReflectionEntry] = {}
        self._repo = repo
        if self._repo is not None:
            self._load_all()

    # -- Persistence --------------------------------------------------------

    def _load_all(self) -> None:
        try:
            rows = self._repo.load_all_reflections()
        except Exception:
            return
        for data in rows:
            try:
                entry = ReflectionEntry.from_dict(data)
                self._entries[entry.entry_id] = entry
            except Exception:
                pass

    def _persist(self, entry: ReflectionEntry) -> None:
        if self._repo is None:
            return
        try:
            self._repo.save_reflection(entry.to_dict())
        except Exception:
            pass

    # -- CRUD ---------------------------------------------------------------

    def store(
        self,
        entry_or_game_id: ReflectionEntry | str = "",
        *,
        player_id: str = "",
        role: str = "",
        faction_won: str | bool = "",
        text: str = "",
        tags: list[str] | None = None,
        situation: str | dict | None = None,
    ) -> None:
        """Store a reflection entry. Accepts either a ReflectionEntry or keyword args."""
        if isinstance(entry_or_game_id, ReflectionEntry):
            entry = entry_or_game_id
        else:
            entry = ReflectionEntry(
                entry_id=uuid.uuid4().hex[:12],
                game_id=str(entry_or_game_id),
                player_id=player_id,
                role=role,
                faction_won=bool(faction_won) if isinstance(faction_won, (bool, int)) else faction_won == "werewolf",
                text=text,
                tags=tags or [],
                situation=json.dumps(situation, ensure_ascii=False) if isinstance(situation, dict) else str(situation or ""),
            )
        self._entries[entry.entry_id] = entry
        self._persist(entry)

    def get(self, entry_id: str) -> ReflectionEntry | None:
        return self._entries.get(entry_id)

    def all_entries(self) -> list[ReflectionEntry]:
        return list(self._entries.values())

    def count(self) -> int:
        return len(self._entries)

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            if self._repo is not None:
                try:
                    self._repo.delete_reflection(entry_id)
                except Exception:
                    pass
            return True
        return False

    # -- Query --------------------------------------------------------------

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
