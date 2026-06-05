"""Long-term reflection memory: post-game unstructured experience storage.

Design doc §10: unstructured reflections (e.g. "last time I trusted
emotional speech and got misaligned") go into long-term memory.
Vote chains, claims, and attack/defense logic stay structured in RelationGraph.

This module stores reflections with tag-based retrieval. Vector search
is a future extension — current implementation uses exact tag/role matching.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from werewolf_agent.memory.schemas import CrossGameQuery, ReflectionEntry


_LOG = logging.getLogger(__name__)


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

    def _persist(self, entry: ReflectionEntry, raise_on_failure: bool = False) -> None:
        if self._repo is None:
            return
        try:
            self._repo.save_reflection(entry.to_dict())
        except Exception:
            # MEM-25: by default, keep the legacy silent-on-failure
            # behavior so production callers don't start seeing new
            # exceptions after a dependency upgrade. Tests /
            # migration scripts can opt in via raise_on_failure=True
            # to surface the underlying error.
            if raise_on_failure:
                raise
            _LOG.warning(
                "Failed to persist reflection %s for player %s",
                entry.entry_id, entry.player_id, exc_info=True,
            )

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
        raise_on_failure: bool = False,
    ) -> None:
        """Store a reflection entry. Accepts either a ReflectionEntry or keyword args.

        MEM-25: ``raise_on_failure`` is forwarded to ``_persist``;
        when True, a DB write failure propagates out of ``store``
        instead of being logged. Default is False to preserve the
        legacy silent-on-failure behavior.
        """
        if isinstance(entry_or_game_id, ReflectionEntry):
            entry = entry_or_game_id
        else:
            # MEM-NEW-7: faction_won MUST be a bool. The pre-fix code
            # accepted strings via ``faction_won == "werewolf"``, which
            # silently mapped "true" / "yes" / "win" to False and
            # corrupted the reflection's win/loss record. Drop the
            # string fallback; raise TypeError so the bug surfaces at
            # the call site, not deep inside a cross-game query.
            if not isinstance(faction_won, bool):
                raise TypeError(
                    f"ReflectionMemory.store: faction_won must be bool, "
                    f"got {type(faction_won).__name__}: {faction_won!r}"
                )
            entry = ReflectionEntry(
                entry_id=uuid.uuid4().hex[:12],
                game_id=str(entry_or_game_id),
                player_id=player_id,
                role=role,
                faction_won=faction_won,
                text=text,
                tags=tags or [],
                situation=json.dumps(situation, ensure_ascii=False) if isinstance(situation, dict) else str(situation or ""),
            )
        self._entries[entry.entry_id] = entry
        self._persist(entry, raise_on_failure=raise_on_failure)

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

    def query(
        self,
        query: CrossGameQuery,
        vector_index: Any | None = None,
    ) -> list[ReflectionEntry]:
        """Retrieve reflections matching query criteria.

        P0-M6: when a ``vector_index`` is supplied, the query proceeds
        in two stages.

        1. **Filter** by hard constraints (player_id, role, tags,
           faction_won). These are required equality / membership
           predicates; semantic similarity cannot satisfy them.
        2. **Rank** by cosine similarity against ``query.situation`` (or
           ``query.text``). Entries absent from the index fall back to
           score 0 and are appended after scored entries.

        Without a vector index (or with an empty one) the path is the
        pure exact-match behavior used before P0-M6.
        """
        candidates = self._filter_candidates(query)

        if vector_index is None or not getattr(vector_index, "__len__", lambda: 0)():
            # P0-M6: when situation is set, keep the pre-existing substring
            # match against the *situation* field; otherwise no-op.
            return self._apply_situation_filter(candidates, query)[: query.max_results]

        if hasattr(vector_index, "similarity"):
            query_text = query.situation or ""
            scores = vector_index.similarity(query_text)
        else:  # pragma: no cover - defensive: unsupported index impl
            # MEM-13: warn the caller — without a similarity method
            # every entry scores 0 and falls to the unindexed tail,
            # so a "vector search" silently degrades to exact-match
            # order. Make the fallback loud so the upstream caller
            # can fix the index wiring.
            _LOG.warning("vector index has no similarity method, falling back")
            scores = {}

        scored: list[tuple[float, int, ReflectionEntry]] = []
        unindexed: list[ReflectionEntry] = []
        for idx, entry in enumerate(candidates):
            score = float(scores.get(entry.entry_id, 0.0))
            if score > 0.0 or entry.entry_id in scores:
                scored.append((score, -idx, entry))  # tie-break: original order
            else:
                unindexed.append(entry)
        # Higher similarity first; preserve original order on ties.
        scored.sort(key=lambda triple: (-triple[0], triple[1]))
        ranked = [entry for _, _, entry in scored] + unindexed
        return ranked[: query.max_results]

    def _filter_candidates(self, query: CrossGameQuery) -> list[ReflectionEntry]:
        """Apply exact-match *filters* (player_id, role, tags, faction_won).

        MEM-06: also sort by game_id descending (newest first) so the
        truncation step in ``query`` keeps the most recent experience
        when the candidate set is larger than ``max_results``.
        """
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
        if query.faction_won is not None:
            results = [e for e in results if e.faction_won == query.faction_won]
        # Newest first: sort by game_id descending. ``game_id`` is a
        # string; for game ids of the form ``g123`` lex order matches
        # numeric order once the prefix is aligned. For arbitrary ids,
        # reverse-sort is still a stable proxy for "newer first" when
        # ids are timestamp-derived.
        results.sort(key=lambda e: e.game_id, reverse=True)
        return results

    def _apply_situation_filter(
        self,
        candidates: list[ReflectionEntry],
        query: CrossGameQuery,
    ) -> list[ReflectionEntry]:
        if not query.situation:
            return list(candidates)
        return [
            e for e in candidates
            if query.situation.lower() in e.situation.lower()
        ]

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
