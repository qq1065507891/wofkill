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
        reranker: Any | None = None,
    ) -> None:
        self._repository = repository
        self._vector_store = vector_store
        self._seed_provider = seed_provider
        # R1: the reranker is wired into every StrategyRetriever the service
        # builds; defaulting to None preserves the rule-based-only path.
        self._reranker = reranker
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

        # R2: when a vector store is wired, _vector_candidates now
        # returns (score, entry) tuples so the BGE-m3 semantic signal
        # reaches the StrategyRetriever instead of being discarded
        # after filtering.
        candidate_entries: list[RAGEntry] = entries
        vector_scores: dict[str, float] | None = None
        if self._vector_store is not None:
            scored = self._vector_candidates(query, entries)
            candidate_entries = [entry for _, entry in scored]
            # Build the entry_id → score map for the retriever's merge
            # step. N6: the original docstring claimed "entries from
            # the metadata fallback path carry score 0.0" but after R2
            # the ``if score > 0.0`` filter drops those entries from
            # the merge map entirely — the fallback path is therefore
            # treated as "no vector signal" by the retriever (which
            # then uses pure rule-based ranking for them). The filter
            # is intentional: keeping fallback entries in the map
            # with a 0.0 score would pull their merged relevance down
            # for no semantic reason.
            vector_scores = {
                entry.entry_id: score for score, entry in scored if score > 0.0
            }

        injector = RAGInjector(
            StrategyRetriever(
                candidate_entries,
                reranker=self._reranker,
                vector_scores=vector_scores,
            )
        )
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
    ) -> list[tuple[float, RAGEntry]]:
        """Return (vector_score, entry) candidates for the StrategyRetriever.

        R2: the vector score (when available) is propagated alongside the
        entry so the retriever can fold it into the final rank. Entries
        admitted via the metadata fallback path (no vector hit) carry
        score 0.0 — the retriever treats those as "no vector signal" and
        falls back to pure rule-based ranking.
        """
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
        selected: dict[str, tuple[float, RAGEntry]] = {}
        try:
            vector_results = self._vector_store.query(query_text, top_k=max(query.max_results * 6, 20))
            # Normalize raw vector scores to [0,1] so the merge math in
            # the retriever stays in the same scale as the rule-based
            # score. The LocalVectorStore can emit scores > 1.0 because
            # IDF is unbounded; a hash-based embedding store could emit
            # negative similarities. Divide each by the per-call max so
            # the top hit anchors at 1.0 and ordering is preserved.
            raw_scores = [float(r.get("score", 0.0)) for r in vector_results]
            score_max = max((s for s in raw_scores if s > 0.0), default=1.0)
            for result, raw in zip(vector_results, raw_scores):
                entry = by_id.get(result.get("doc_id"))
                if entry is None:
                    continue
                normalized = max(0.0, min(1.0, raw / score_max)) if score_max > 0 else 0.0
                selected[entry.entry_id] = (normalized, entry)
        except Exception:
            logger.warning("Vector RAG query failed; using metadata candidates", exc_info=True)

        for entry in entries:
            meta = entry.metadata
            if query.ruleset_id and meta.ruleset_id and meta.ruleset_id != query.ruleset_id:
                continue
            # R9: the previous version used parallel ``if`` checks
            # (OR semantics), so a cross-role case slipped into the
            # candidate pool whenever the phase happened to match.
            # We now require BOTH role and phase to match (each side
            # accepts the ``general`` wildcard for universal entries),
            # which preserves role isolation across the metadata
            # fallback path.
            role_ok = (
                not query.role
                or meta.role_perspective in (query.role, "general", "")
            )
            phase_ok = (
                not query.phase
                or meta.phase in (query.phase, "general", "")
            )
            if role_ok and phase_ok:
                selected.setdefault(entry.entry_id, (0.0, entry))

        if selected:
            return list(selected.values())
        # Final fallback: no vector hit and no metadata match. Return all
        # entries with score 0.0 so the retriever's rule-based path can
        # still produce ordering.
        return [(0.0, entry) for entry in entries]

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
