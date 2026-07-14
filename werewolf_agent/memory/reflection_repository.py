# -*- coding: utf-8 -*-
"""
存储和检索跨局反思记忆。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-14

使用示例:
    >>> from werewolf_agent.memory.reflection_repository import ReflectionMemory
    >>> ReflectionMemory()
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import uuid
from collections import Counter
from typing import Any

from werewolf_agent.memory.schemas import (
    CrossGameQuery,
    ReflectionEntry,
    ReflectionEntryV2,
    ReflectionQualityStatus,
)
from werewolf_agent.memory.reflection_sanitization import anonymize_player_ids_recursive

_LOG = logging.getLogger("werewolf_agent.memory.reflection")


class ReflectionMemory:
    """Stores and retrieves post-game reflections across games.

    When a ``repo`` (PostgresGameRepository) is provided, reflections are
    persisted to the ``reflections`` table.  Otherwise entries are kept
    in memory only (suitable for tests and single-run scenarios).
    """

    def __init__(self, repo: Any | None = None) -> None:
        self._entries: dict[str, ReflectionEntry] = {}
        self._v2_entries: dict[str, ReflectionEntryV2] = {}
        self._view_id_key = secrets.token_bytes(32)
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
            # 失败事务留下的隔离行不得进入跨局提示；缺少该字段的历史行保持兼容。
            if data.get("_persistence_active") is False:
                continue
            payload = {
                key: value for key, value in data.items()
                if key != "_persistence_active"
            }
            try:
                if payload.get("schema_version") == 2:
                    entry_v2 = ReflectionEntryV2.from_dict(payload)
                    self._v2_entries[entry_v2.entry_id] = entry_v2
                else:
                    entry = ReflectionEntry.from_dict(payload)
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

    def _persist_v2(
        self,
        entry: ReflectionEntryV2,
        raise_on_failure: bool = False,
    ) -> bool:
        if self._repo is None:
            return True
        try:
            self._repo.save_reflection(entry.to_dict())
            return True
        except Exception:
            if raise_on_failure:
                raise
            _LOG.warning(
                "Failed to persist V2 reflection %s for player %s",
                entry.entry_id, entry.player_id, exc_info=True,
            )
            return False

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

    def store_v2(
        self,
        entry: ReflectionEntryV2,
        *,
        raise_on_failure: bool = False,
    ) -> None:
        if self._persist_v2(entry, raise_on_failure=raise_on_failure):
            self._v2_entries[entry.entry_id] = entry

    def get(self, entry_id: str) -> ReflectionEntry | None:
        return self._entries.get(entry_id)

    def all_entries(self) -> list[ReflectionEntry]:
        return list(self._entries.values())

    def all_v2_entries(self) -> list[ReflectionEntryV2]:
        return list(self._v2_entries.values())

    def restore_v2_local(
        self,
        entry_id: str,
        previous: ReflectionEntryV2 | None,
    ) -> None:
        """仅恢复进程内 V2 行；仓储补偿由事务编排器单独负责。"""
        if previous is None:
            self._v2_entries.pop(entry_id, None)
        else:
            self._v2_entries[entry_id] = previous

    def count(self) -> int:
        return len(self._entries) + len(self._v2_entries)

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            if self._repo is not None:
                try:
                    self._repo.delete_reflection(entry_id)
                except Exception:
                    pass
            return True
        if entry_id in self._v2_entries:
            del self._v2_entries[entry_id]
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

        # MEM-NEW-11: an object with __len__ but no similarity method
        # is not a valid vector index. Catch it HERE, in the guard,
        # so the rest of the function never sees it — the prior
        # post-guard ``hasattr`` check at the call site was a
        # defensive but easy-to-miss fallback, and an object with a
        # `similarity` attribute that raises on call would have
        # leaked through with an AttributeError deep inside the
        # rank loop. The contract for a valid index is BOTH
        # __len__ AND similarity; missing either one falls
        # through to the exact-match path with a warning.
        if (
            vector_index is None
            or not getattr(vector_index, "__len__", lambda: 0)()
            or not hasattr(vector_index, "similarity")
        ):
            if vector_index is not None and not hasattr(vector_index, "similarity"):
                # MEM-13: warn the caller — without a similarity
                # method every entry scores 0 and falls to the
                # unindexed tail, so a "vector search" silently
                # degrades to exact-match order. Make the fallback
                # loud so the upstream caller can fix the index
                # wiring.
                _LOG.warning(
                    "vector index has no similarity method, falling back"
                )
            # P0-M6: when situation is set, keep the pre-existing substring
            # match against the *situation* field; otherwise no-op.
            return self._apply_situation_filter(candidates, query)[: query.max_results]

        query_text = query.situation or ""
        scores = vector_index.similarity(query_text)

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

    def query_live(self, query: CrossGameQuery) -> list[ReflectionEntryV2]:
        """Retrieve approved V2 reflections eligible for live prompts."""
        results = [
            entry for entry in self._v2_entries.values()
            if entry.quality_status == ReflectionQualityStatus.APPROVED
        ]
        if query.player_id:
            results = [e for e in results if e.player_id == query.player_id]
        if query.role:
            results = [e for e in results if e.role == query.role]
        if query.faction_won is not None:
            results = [e for e in results if e.faction_won == query.faction_won]
        if query.tags:
            results = [
                e for e in results
                if any(self._v2_has_tag(e, tag) for tag in query.tags)
            ]
        if query.situation:
            needle = query.situation.lower()
            results = [
                e for e in results
                if needle in json.dumps(e.to_dict(), ensure_ascii=False).lower()
            ]
        results.sort(key=self._live_rank_key)
        return [self._cross_game_view(entry) for entry in results[: query.max_results]]

    def _cross_game_view(self, entry: ReflectionEntryV2) -> ReflectionEntryV2:
        """返回保留聚合字段的匿名化视图，并使用本次运行专属 opaque ID。"""
        payload = anonymize_player_ids_recursive(entry.model_dump(mode="json"))
        identity = f"{entry.game_id}\0{entry.entry_id}\0{entry.player_id}".encode("utf-8")
        digest = hmac.new(self._view_id_key, identity, hashlib.sha256).hexdigest()[:24]
        payload["entry_id"] = f"view_{digest}"
        payload["player_id"] = "历史玩家本人"
        payload["source"]["llm_self_review"] = ""
        payload["source"]["auto_review_summary"] = ""
        return ReflectionEntryV2.model_validate(payload)

    @staticmethod
    def _v2_has_tag(entry: ReflectionEntryV2, tag: str) -> bool:
        fields = [
            entry.role,
            entry.faction,
            *entry.situation_signature.phase_focus,
            *entry.situation_signature.game_patterns,
            *(p.category for p in entry.mistake_patterns),
        ]
        return tag in fields

    @staticmethod
    def _live_rank_key(entry: ReflectionEntryV2) -> tuple[float, str, str]:
        return (-entry.quality_score, str(entry.game_id), str(entry.entry_id))

    def live_error_pattern(
        self,
        player_id: str,
        role: str = "",
    ) -> dict[str, Any]:
        entries = self.query_live(CrossGameQuery(
            player_id=player_id,
            role=role,
            max_results=100,
        ))
        counter: Counter[str] = Counter()
        preserved = 0
        strength_labels: list[str] = []
        for entry in entries:
            for pattern in entry.mistake_patterns:
                if pattern.category:
                    counter[pattern.category] += 1
            if entry.preserved_strengths:
                preserved += 1
                for strength in entry.preserved_strengths[:1]:
                    if strength.category:
                        strength_labels.append(strength.category)
        total_mistakes = sum(counter.values())
        dominant = (
            round(counter.most_common(1)[0][1] / total_mistakes, 2)
            if total_mistakes and counter
            else 0.0
        )
        return {
            "top_mistakes": counter.most_common(2),
            "preserved_strength_count": preserved,
            "preserved_strength_labels": strength_labels[:2],
            "total_reflections": len(entries),
            "same_role_reflections": sum(1 for e in entries if e.role == role),
            "dominant_mistake_ratio": dominant,
            "current_role": role,
        }
