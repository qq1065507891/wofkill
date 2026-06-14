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
from werewolf_agent.rag.schemas import RAGEntry, RAGHit, RAGQuery, VisibilityBoundary
from werewolf_agent.rag.tactical_text import build_rag_retrieval_text

logger = logging.getLogger(__name__)


# G-R4-03: absolute floor for raw vector scores. Hits below this
# value are treated as "no real signal" and are dropped from the
# candidate pool before the per-call max normalization runs.
# Without this floor the per-call max normalization promoted the
# weakest top hit to 1.0, which then pulled the merge formula
# above the rule-only path for any unrelated entry the vector
# store happened to return. Empirically the LocalVectorStore's
# TF-IDF score is in [0, log(N)+1] for a corpus of N documents
# with at least one shared token; a value of 0.1 keeps the floor
# well below any meaningful single-token hit and well above the
# 0.0 default for entries the store never scored.
_WEAK_VECTOR_THRESHOLD: float = 0.1


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
        # G-R4-13: hold a long-lived RAGInjector so the audit log
        # accumulates across calls (deque maxlen caps the total).
        # Pre-fix: a fresh RAGInjector was built on every
        # ``retrieve_live_hints`` call, so the audit log was a
        # 1-entry deque and the previous turn's audit record was
        # lost the moment the next call returned. The audit log is
        # the only observability surface for "why did this hit
        # surface for this player" — losing it after every turn
        # defeated the purpose.
        #
        # The retriever and (if used) vector scores are wired
        # per-call via ``_refresh_injector`` so the new candidate
        # pool reaches the retriever; the injector's audit deque
        # itself is the long-lived part.
        self._injector: RAGInjector | None = None

    def _ensure_injector(self) -> RAGInjector:
        """Return the long-lived RAGInjector, creating it on first use."""
        if self._injector is None:
            self._injector = RAGInjector(
                StrategyRetriever([]),
            )
        return self._injector

    def _refresh_injector(
        self,
        candidate_entries: list[RAGEntry],
        vector_scores: dict[str, float] | None,
    ) -> RAGInjector:
        """Swap the retriever inside the long-lived injector so the
        new candidate pool + vector scores reach the per-call
        StrategyRetriever while the audit deque keeps accumulating.
        """
        injector = self._ensure_injector()
        injector._retriever = StrategyRetriever(
            candidate_entries,
            reranker=self._reranker,
            vector_scores=vector_scores,
        )
        return injector

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

        # G-R4-13: reuse the long-lived injector's retriever so the
        # audit deque keeps accumulating. A new StrategyRetriever is
        # wired in via ``_refresh_injector`` for this call's
        # candidate pool + vector scores; the injector's audit log
        # and last_audit survive across calls.
        injector = self._refresh_injector(
            candidate_entries,
            vector_scores,
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

        P0-G1/RAG V2: only title plus prompt-safe tactical-frame fields
        are surfaced.
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

    def _passes_live_metadata_filter(self, query: RAGQuery, entry: RAGEntry) -> bool:
        """Hard metadata gate shared by vector hits and metadata fallback."""
        meta = entry.metadata
        if meta.visibility_boundary not in (
            VisibilityBoundary.PUBLIC_ONLY,
            VisibilityBoundary.PLAYER_PERSPECTIVE,
        ):
            return False
        if query.ruleset_id and meta.ruleset_id and meta.ruleset_id != query.ruleset_id:
            return False
        role_ok = (
            not query.role
            or meta.role_perspective in (query.role, "general", "any", "")
        )
        phase_ok = (
            not query.phase
            or meta.phase in (query.phase, "general", "")
        )
        return role_ok and phase_ok

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
            #
            # G-R4-03: also drop hits whose absolute raw score is below
            # ``_WEAK_VECTOR_THRESHOLD``. The per-call max normalization
            # alone promoted the weakest top hit to 1.0, which then
            # pulled the merge formula above the rule-only path for
            # any unrelated entry the vector store happened to return.
            # The threshold gives the retriever a documented
            # "no real signal" floor that keeps the rule path in
            # charge when the vector recall is empty / noisy.
            eligible_results: list[tuple[dict[str, Any], float, RAGEntry]] = []
            for result in vector_results:
                entry = by_id.get(result.get("doc_id"))
                if entry is None:
                    continue
                if not self._passes_live_metadata_filter(query, entry):
                    continue
                raw = float(result.get("score", 0.0))
                if raw < _WEAK_VECTOR_THRESHOLD:
                    # Below the absolute floor: the vector store had
                    # nothing meaningful to say about this doc, so
                    # don't let a 1.0-normalized value compete with
                    # a clean rule-only path.
                    continue
                eligible_results.append((result, raw, entry))
            score_max = max((raw for _, raw, _ in eligible_results if raw > 0.0), default=1.0)
            for _result, raw, entry in eligible_results:
                normalized = max(0.0, min(1.0, raw / score_max)) if score_max > 0 else 0.0
                selected[entry.entry_id] = (normalized, entry)
        except Exception:
            logger.warning("Vector RAG query failed; using metadata candidates", exc_info=True)

        for entry in entries:
            if not self._passes_live_metadata_filter(query, entry):
                continue
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
            #
            # G-R4-02: the role check also accepts ``"any"`` as a
            # universal-perspective wildcard, matching the convention
            # used by the ``基础常识`` seed family (金水 / 银水 /
            # 对跳判断 / 警徽票权重, etc.). Pre-fix the filter only
            # accepted ``(query.role, "general", "")``, which
            # silently dropped every "any" entry once at least one
            # other entry had populated the ``selected`` pool.
            role_ok = (
                not query.role
                or meta.role_perspective in (query.role, "general", "any", "")
            )
            phase_ok = (
                not query.phase
                or meta.phase in (query.phase, "general", "")
            )
            if role_ok and phase_ok:
                selected.setdefault(entry.entry_id, (0.0, entry))

        if selected:
            return list(selected.values())
        # Final fallback: no vector hit and no metadata match. Preserve
        # the hard metadata boundary even when returning score-0 entries
        # for the retriever's rule-based ordering.
        return [
            (0.0, entry)
            for entry in entries
            if self._passes_live_metadata_filter(query, entry)
        ]

    def _index_entry(self, entry: RAGEntry) -> None:
        text = build_rag_retrieval_text(entry, max_chars=1500)
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
