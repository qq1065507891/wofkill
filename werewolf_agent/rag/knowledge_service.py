"""Unified RAG knowledge service.

The service keeps curated seed knowledge usable in every deployment mode:
in-memory local runs, SQLite/PostgreSQL storage, and pgvector-backed recall.
Runtime callers should depend on this service instead of knowing whether the
knowledge came from code seeds, repository storage, or vector search.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from werewolf_agent.rag.ingestion import create_seed_entries
from werewolf_agent.rag.injector import InjectionContext, RAGInjector
from werewolf_agent.rag.persistence import load_rag_entries, save_rag_entries
from werewolf_agent.rag.retriever import StrategyRetriever
from werewolf_agent.rag.schemas import RAGEntry, RAGHit, RAGQuery

logger = logging.getLogger(__name__)


class RAGKnowledgeService:
    """Coordinates seed, repository, vector, and live-safe RAG retrieval."""

    def __init__(
        self,
        *,
        repository: Any | None = None,
        vector_store: Any | None = None,
        seed_provider: Callable[[], list[RAGEntry]] = create_seed_entries,
    ) -> None:
        self._repository = repository
        self._vector_store = vector_store
        self._seed_provider = seed_provider
        self._seed_entries: list[RAGEntry] | None = None
        self._entries_cache: dict[str, RAGEntry] | None = None
        self._last_audit: Any | None = None

    def ensure_seeded(self) -> dict[str, int]:
        """Upsert seed entries into repository and vector store when available."""
        seeds = self._seeds()
        saved_count = 0
        indexed_count = 0

        if self._repository is not None and hasattr(self._repository, "save_rag_entries"):
            data = save_rag_entries(seeds)
            self._repository.save_rag_entries(data)
            saved_count = len(data)

        if self._vector_store is not None:
            for entry in seeds:
                self._index_entry(entry)
                indexed_count += 1

        self._entries_cache = None
        return {
            "seed_count": len(seeds),
            "saved_count": saved_count,
            "indexed_count": indexed_count,
        }

    def retrieve_live_hints(
        self,
        query: RAGQuery,
        *,
        game_id: str = "",
        player_id: str = "",
    ) -> list[RAGHit]:
        """Retrieve strategy hints that are safe for live-player context."""
        entries = self._load_entries()
        if not entries:
            return []

        candidate_entries = entries
        if self._vector_store is not None:
            candidate_entries = self._vector_candidates(query, entries)

        injector = RAGInjector(StrategyRetriever(candidate_entries))
        hits = injector.inject(
            query,
            injection_context=InjectionContext.LIVE_PLAYER,
            game_id=game_id,
            player_id=player_id,
        )
        self._last_audit = injector.last_audit()
        return hits

    def hits_to_context_items(
        self,
        hits: list[RAGHit],
        max_items: int = 3,
    ) -> list[dict[str, Any]]:
        """Convert RAG hits to full audit-aware context items.

        This path retains relevance, quality, source type, visibility,
        and the display annotation. Use it for audit logs, moderator
        views, and review tooling — not for the live player prompt.
        """
        return RAGInjector.hits_to_context_items(
            hits,
            max_items=max_items,
        )

    def hits_to_prompt_lines(
        self,
        hits: list[RAGHit],
        max_items: int = 3,
    ) -> list[dict[str, Any]]:
        """Convert RAG hits to slim prompt lines for the live player.

        P0-G1: only title, summary, and 2-3 key_decisions are surfaced.
        See :func:`werewolf_agent.rag.prompt_renderer.hits_to_prompt_lines`
        for the full exclusion list.
        """
        return RAGInjector.hits_to_prompt_lines(
            hits,
            max_items=max_items,
        )

    def last_audit(self) -> Any | None:
        return self._last_audit

    def _seeds(self) -> list[RAGEntry]:
        if self._seed_entries is None:
            self._seed_entries = self._seed_provider()
        return list(self._seed_entries)

    def _load_entries(self) -> list[RAGEntry]:
        if self._entries_cache is not None:
            return list(self._entries_cache.values())

        entries: list[RAGEntry] = []
        if self._repository is not None and hasattr(self._repository, "load_rag_entries"):
            try:
                data = self._repository.load_rag_entries()
                entries = load_rag_entries(data)
                if not entries:
                    self.ensure_seeded()
                    data = self._repository.load_rag_entries()
                    entries = load_rag_entries(data)
            except Exception:
                logger.warning("Failed to load RAG entries from repository; using seeds", exc_info=True)
                entries = []

        if not entries:
            entries = self._seeds()

        self._entries_cache = {entry.entry_id: entry for entry in entries}
        return list(self._entries_cache.values())

    def _vector_candidates(
        self,
        query: RAGQuery,
        entries: list[RAGEntry],
    ) -> list[RAGEntry]:
        by_id = {entry.entry_id: entry for entry in entries}
        query_text = " ".join(
            part for part in (
                query.role,
                query.phase,
                query.situation,
                query.ruleset_id,
            )
            if part
        )
        selected: dict[str, RAGEntry] = {}
        try:
            vector_results = self._vector_store.query(query_text, top_k=max(query.max_results * 6, 20))
            for result in vector_results:
                entry = by_id.get(result.get("doc_id"))
                if entry is not None:
                    selected[entry.entry_id] = entry
        except Exception:
            logger.warning("Vector RAG query failed; using metadata candidates", exc_info=True)

        for entry in entries:
            meta = entry.metadata
            if query.ruleset_id and meta.ruleset_id and meta.ruleset_id != query.ruleset_id:
                continue
            if query.role and meta.role_perspective in (query.role, "general"):
                selected[entry.entry_id] = entry
            if query.phase and meta.phase in (query.phase, "general"):
                selected[entry.entry_id] = entry

        return list(selected.values()) if selected else entries

    def _index_entry(self, entry: RAGEntry) -> None:
        text = "\n".join([
            entry.title,
            entry.summary,
            "\n".join(entry.key_decisions),
            " ".join(entry.metadata.tags),
        ])
        metadata = {
            "entry_id": entry.entry_id,
            "ruleset_id": entry.metadata.ruleset_id,
            "phase": entry.metadata.phase,
            "role_perspective": entry.metadata.role_perspective,
            "quality_grade": entry.metadata.quality_grade.value,
            "review_status": entry.metadata.review_status.value,
            "visibility_boundary": entry.metadata.visibility_boundary.value,
            "source_type": entry.metadata.source.source_type.value,
        }
        self._vector_store.add(entry.entry_id, text, metadata)
