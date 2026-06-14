"""Tests for the slim RAG prompt renderer (P0-G1 + P0-G2).

P0-G1: The slim renderer must only include ``title``, ``summary``, and
``key_decisions`` (capped at 3) — never the audit-only metadata that
``RAGInjector.hits_to_context_items()`` exposes.

P0-G2: Specifically, the live prompt must NOT contain
``relevance_score``, ``quality_grade.value``, ``source_type.value``,
``visibility_boundary.value``, or ``display_annotation``. Those are
audit-only fields; they belong in ``RAGInjector.audit_log()`` and
``RAGHit``, not in the player-facing prompt.

The renderer is a tiny pure function on purpose so the tests can
construct ``RAGHit`` instances directly and assert on the rendered
output without touching the retriever, injector, or any IO.
"""

from __future__ import annotations

import json

import pytest

from werewolf_agent.rag.prompt_renderer import (
    _FORBIDDEN_LIVE_FIELDS,
    _MAX_KEY_DECISIONS_IN_PROMPT,
    dedup_hits_by_similarity,
    hits_to_prompt_lines,
    hits_to_prompt_lines_json,
    render_hit_for_prompt,
)
from werewolf_agent.rag.schemas import (
    CaseType,
    QualityGrade,
    RAGHit,
    SourceType,
    VisibilityBoundary,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_hit(
    *,
    entry_id: str = "ext_001",
    title: str = "京城大师赛 250415 抗推预言家",
    summary: str = "狼队在白天通过抗推预言家获得票数优势。",
    relevance: float = 0.83,
    quality: QualityGrade = QualityGrade.HIGH_RANK_GAME,
    source: SourceType = SourceType.PUBLIC_TOURNAMENT,
    visibility: VisibilityBoundary = VisibilityBoundary.PLAYER_PERSPECTIVE,
    key_decisions: list[str] | None = None,
    annotation: str = "[public_tournament|high_rank_game]",
) -> RAGHit:
    """Build a realistic RAGHit for renderer unit tests."""
    return RAGHit(
        entry_id=entry_id,
        title=title,
        summary=summary,
        relevance_score=relevance,
        quality_grade=quality,
        source_type=source,
        visibility_boundary=visibility,
        case_type=CaseType.EXTERNAL_HIGH_END_CASE,
        role_perspective="werewolf",
        phase="speech",
        key_decisions=key_decisions
        if key_decisions is not None
        else [
            "白天全力归票预言家",
            "预言家抗推后改换身份打深钩",
            "夜里优先解神牌",
        ],
        short_quotes=["p04 票型抱团"],
        tags=["抗推", "预言家", "神牌"],
        allowed_in_live_context=True,
        display_annotation=annotation,
    )


# ===================================================================
# P0-G1: slim renderer drops audit metadata
# ===================================================================


class TestPromptRenderDropsMetadata:
    """P0-G1: The slim renderer must only expose title/summary/key_decisions."""

    def test_render_hit_for_prompt_has_only_three_keys(self) -> None:
        hit = _make_hit()
        line = render_hit_for_prompt(hit)
        # R3: the slim renderer adds a ``type`` discriminator so the
        # context layer can identify and clear previous rag_hit items
        # between turns. The four keys are: type, title, summary,
        # key_decisions.
        assert set(line.keys()) == {"type", "title", "summary", "key_decisions"}

    def test_render_hit_for_prompt_includes_type_field(self) -> None:
        """R3: every slim prompt line must carry ``type == "rag_hit"`` so
        the context layer's filter
        ``[item for item in ctx.rag_hints if item.get("type") != "rag_hit"]``
        actually drops previous rag hits. Without the discriminator
        the filter is a no-op and old slim items pile up across turns.
        """
        hit = _make_hit()
        line = render_hit_for_prompt(hit)
        assert line.get("type") == "rag_hit", (
            f"R3: slim line must carry type='rag_hit'; got {line!r}"
        )

    def test_render_hit_preserves_title_summary(self) -> None:
        hit = _make_hit(
            title="京城大师赛 250415 抗推预言家",
            summary="狼队在白天通过抗推预言家获得票数优势。",
        )
        line = render_hit_for_prompt(hit)
        assert line["title"] == "京城大师赛 250415 抗推预言家"
        assert line["summary"] == "狼队在白天通过抗推预言家获得票数优势。"

    def test_render_hit_truncates_key_decisions_to_three(self) -> None:
        hit = _make_hit(
            key_decisions=[
                "决策1: 白天全力归票预言家",
                "决策2: 预言家抗推后改换身份打深钩",
                "决策3: 夜里优先解神牌",
                "决策4: 不再深钩（不应当出现）",
                "决策5: 跳预言家时自曝（不应当出现）",
            ],
        )
        line = render_hit_for_prompt(hit)
        assert len(line["key_decisions"]) == _MAX_KEY_DECISIONS_IN_PROMPT
        assert line["key_decisions"] == [
            "决策1: 白天全力归票预言家",
            "决策2: 预言家抗推后改换身份打深钩",
            "决策3: 夜里优先解神牌",
        ]

    def test_render_hit_with_zero_key_decisions(self) -> None:
        hit = _make_hit(key_decisions=[])
        line = render_hit_for_prompt(hit)
        assert line["key_decisions"] == []

    def test_hits_to_prompt_lines_respects_max_items(self) -> None:
        # P1-G5: distinct titles + distinct summaries with zero token
        # overlap so the dedup pass doesn't collapse them.
        hits = [
            _make_hit(
                entry_id=f"ext_{i:03d}",
                title=f"甲{i}",
                summary=f"乙{i}",
            )
            for i in range(5)
        ]
        lines = hits_to_prompt_lines(hits, max_items=2)
        assert len(lines) == 2
        assert lines[0]["title"] == "甲0"
        assert lines[1]["title"] == "甲1"

    def test_hits_to_prompt_lines_default_max_is_three(self) -> None:
        # G-R4-09: the dedup cap is now caller-controlled. The
        # default of ``hits_to_prompt_lines`` is ``max_items=3``
        # (matching the live-injection cap), and the public helper
        # honors it. With 5 distinct hits (no near-duplicates), 3
        # are returned.
        hits = [
            _make_hit(
                entry_id=f"ext_{i:03d}",
                title=f"甲{i}",
                summary=f"乙{i}",
            )
            for i in range(5)
        ]
        lines = hits_to_prompt_lines(hits)
        assert len(lines) == 3

    def test_hits_to_prompt_lines_empty(self) -> None:
        assert hits_to_prompt_lines([]) == []


# ===================================================================
# P0-G2: live prompt must not include audit-only fields
# ===================================================================


class TestNoMetadataInLivePrompt:
    """P0-G2: Audit-only fields must NEVER leak into the live prompt."""

    def test_render_hit_omits_relevance_score(self) -> None:
        line = render_hit_for_prompt(_make_hit(relevance=0.83))
        assert "relevance_score" not in line
        assert "relevance" not in line

    def test_render_hit_omits_quality_grade(self) -> None:
        line = render_hit_for_prompt(
            _make_hit(quality=QualityGrade.HIGH_RANK_GAME),
        )
        assert "quality_grade" not in line
        assert "quality" not in line

    def test_render_hit_omits_source_type(self) -> None:
        line = render_hit_for_prompt(
            _make_hit(source=SourceType.PUBLIC_TOURNAMENT),
        )
        assert "source_type" not in line

    def test_render_hit_omits_visibility_boundary(self) -> None:
        line = render_hit_for_prompt(
            _make_hit(visibility=VisibilityBoundary.PLAYER_PERSPECTIVE),
        )
        assert "visibility" not in line
        assert "visibility_boundary" not in line

    def test_render_hit_omits_display_annotation(self) -> None:
        line = render_hit_for_prompt(
            _make_hit(annotation="[public_tournament|high_rank_game]"),
        )
        assert "display_annotation" not in line
        assert "annotation" not in line

    def test_render_hit_omits_case_type_role_perspective_tags(self) -> None:
        line = render_hit_for_prompt(_make_hit())
        for forbidden in (
            "case_type",
            "role_perspective",
            "tags",
            "short_quotes",
            "allowed_in_live",
            "allowed_in_live_context",
            "phase",
            "entry_id",
        ):
            assert forbidden not in line, (
                f"Audit-only field {forbidden!r} leaked into live prompt line"
            )

    def test_forbidden_live_fields_constant_covers_all_leak_risks(self) -> None:
        """Sanity check: the constant must include the P0-G2 must-not-include list."""
        must_be_forbidden = {
            "relevance_score",
            "relevance",
            "quality_grade",
            "quality",
            "source_type",
            "visibility",
            "visibility_boundary",
            "display_annotation",
            "annotation",
            "allowed_in_live",
            "allowed_in_live_context",
        }
        assert must_be_forbidden.issubset(_FORBIDDEN_LIVE_FIELDS), (
            f"_FORBIDDEN_LIVE_FIELDS must cover {must_be_forbidden - _FORBIDDEN_LIVE_FIELDS}"
        )

    def test_no_metadata_leaks_through_json_serialization(self) -> None:
        """End-to-end: when the slim lines are JSON-encoded, no audit field
        value is recoverable from the string. This guards against someone
        adding a new key to ``render_hit_for_prompt`` and forgetting the
        JSON path.
        """
        hit = _make_hit(
            relevance=0.83,
            quality=QualityGrade.HIGH_RANK_GAME,
            source=SourceType.PUBLIC_TOURNAMENT,
            visibility=VisibilityBoundary.PLAYER_PERSPECTIVE,
            annotation="[public_tournament|high_rank_game]",
        )
        encoded = hits_to_prompt_lines_json([hit], max_items=1)
        decoded = json.loads(encoded)
        # R3: ``type`` is the slim-line discriminator so the context layer
        # can clear previous rag_hit items between turns.
        assert decoded == [{
            "type": "rag_hit",
            "title": hit.title,
            "summary": hit.summary,
            "key_decisions": hit.key_decisions,
        }]
        # None of the audit field names should appear in the JSON string.
        for forbidden_name in (
            "relevance_score",
            "quality_grade",
            "source_type",
            "visibility_boundary",
            "display_annotation",
            "allowed_in_live",
            "case_type",
            "role_perspective",
        ):
            assert forbidden_name not in encoded, (
                f"JSON-serialized live prompt leaks field name {forbidden_name!r}"
            )
        # The actual score / quality value must not appear in the string either.
        assert "0.83" not in encoded
        assert "high_rank_game" not in encoded
        assert "public_tournament" not in encoded
        assert "player_perspective" not in encoded


# ===================================================================
# Cross-check: slim renderer is independent of the audit helper
# ===================================================================


class TestSlimRendererIndependence:
    """The slim renderer must not depend on the audit ``hits_to_context_items``
    implementation. They are independent render paths with different
    audiences (LLM live prompt vs. audit log)."""

    def test_render_hit_is_pure_and_idempotent(self) -> None:
        hit = _make_hit()
        a = render_hit_for_prompt(hit)
        b = render_hit_for_prompt(hit)
        assert a == b
        # Mutating one must not affect the other (list returned, not aliased).
        a["key_decisions"].append("extra")
        assert b["key_decisions"] == [
            "白天全力归票预言家",
            "预言家抗推后改换身份打深钩",
            "夜里优先解神牌",
        ]


# ===================================================================
# P0-G2: slim path vs. audit log — both must coexist correctly
# ===================================================================


class TestSlimAndAuditCoexist:
    """P0-G2: the slim prompt renderer must strip audit metadata, while
    the audit log on the injector retains the full data. Both paths
    must run in the same inject() call without one stomping on the
    other."""

    def test_injector_audit_log_keeps_full_metadata(self) -> None:
        """End-to-end: when the injector runs, the audit log records
        relevance/quality/source/visibility/case_type, even though the
        slim prompt renderer strips them. This is the G2 contract.
        """
        from werewolf_agent.rag.injector import InjectionContext, RAGInjector
        from werewolf_agent.rag.retriever import StrategyRetriever
        from werewolf_agent.rag.schemas import RAGQuery

        hit = _make_hit(
            entry_id="audit_check_001",
            relevance=0.83,
            quality=QualityGrade.HIGH_RANK_GAME,
            source=SourceType.PUBLIC_TOURNAMENT,
            visibility=VisibilityBoundary.PLAYER_PERSPECTIVE,
        )
        # Use a StrategyRetriever built directly from one hit so the
        # test is hermetic (no ingestion / seed dependency).
        from werewolf_agent.rag.schemas import (
            CaseMetadata,
            CaseType,
            RAGEntry,
            ReviewStatus,
            SourceMetadata,
        )

        entry = RAGEntry(
            schema_version=1,
            entry_id="audit_check_001",
            title=hit.title,
            summary=hit.summary,
            key_decisions=hit.key_decisions,
            metadata=CaseMetadata(
                case_type=CaseType.EXTERNAL_HIGH_END_CASE,
                quality_grade=QualityGrade.HIGH_RANK_GAME,
                review_status=ReviewStatus.APPROVED,
                reviewer="test",
                ruleset_id="pre_witch_hunter_idiot_mixed",
                player_count=12,
                phase="speech",
                role_perspective="werewolf",
                visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
                source=SourceMetadata(source_type=SourceType.PUBLIC_TOURNAMENT),
                tags=["抗推"],
            ),
        )
        injector = RAGInjector(StrategyRetriever([entry]))

        hits = injector.inject(
            RAGQuery(role="werewolf", phase="speech"),
            injection_context=InjectionContext.LIVE_PLAYER,
            game_id="g_test",
            player_id="p08",
        )
        # Slim path
        slim_lines = injector.hits_to_prompt_lines(hits, max_items=1)
        assert slim_lines, "slim path must still return the hit"
        # R3: type=rag_hit is the discriminator the context layer uses
        # to clear previous slim items between turns.
        assert set(slim_lines[0].keys()) == {"type", "title", "summary", "key_decisions"}
        assert slim_lines[0]["type"] == "rag_hit"

        # Audit path: full data preserved.
        audit = injector.last_audit()
        assert audit is not None
        assert audit.injection_context == "live_player"
        assert audit.game_id == "g_test"
        assert audit.player_id == "p08"
        assert len(audit.hits) == 1
        audit_hit = audit.hits[0]
        # Full audit payload must include relevance/quality/source/
        # visibility — none of which the slim path exposes.
        assert audit_hit["entry_id"] == "audit_check_001"
        assert audit_hit["relevance_score"] == pytest.approx(0.5, abs=0.5)
        assert audit_hit["quality_grade"] == "high_rank_game"
        assert audit_hit["source_type"] == "public_tournament"
        assert audit_hit["visibility_boundary"] == "player_perspective"
        assert audit_hit["case_type"] == "external_high_end_case"

    def test_slim_renderer_does_not_mutate_hit(self) -> None:
        """Slim rendering is a pure function — the input RAGHit must be
        unchanged after the call, so the audit log can still record the
        full payload."""
        hit = _make_hit()
        before = hit.model_dump()
        _ = render_hit_for_prompt(hit)
        after = hit.model_dump()
        assert before == after


# ===================================================================
# P1-G4: summary truncation relaxed (300 → 800 chars)
# ===================================================================


class TestSummaryTruncation800Chars:
    """P1-G4: The retriever caps summary at 800 chars (was 300). The slim
    renderer must preserve whatever the retriever returned — slim
    rendering does not re-truncate."""

    def test_summary_truncation_800_chars(self) -> None:
        """Build a RAGHit with the new 800-char summary and confirm the
        slim renderer passes it through unchanged."""
        long_summary = "狼" * 800  # exactly 800 chars
        hit = _make_hit(summary=long_summary)
        # Sanity: the test fixture's summary is the 800-char one.
        assert len(hit.summary) == 800

        line = render_hit_for_prompt(hit)
        # Slim renderer does not re-truncate — it trusts the cap at the
        # retriever layer (P1-G4 contract: 800 chars at retriever, not
        # renderer).
        assert line["summary"] == long_summary
        assert len(line["summary"]) == 800


# ===================================================================
# P1-G5: near-duplicate RAG hits merged
# ===================================================================


class TestNearDuplicateHitsMerged:
    """P1-G5: When two RAG hits cover the same tactic, dedup by Jaccard
    similarity on title+summary tokens. Threshold 0.6 — keep the higher
    relevance_score. Cap the final list at 2 hits to keep prompt
    density high."""

    def test_near_duplicate_hits_merged(self) -> None:
        """Two hits that share most of their tokens (>0.6 Jaccard) must
        be deduped; the higher-relevance one is kept."""
        hit_a = _make_hit(
            entry_id="case_a",
            title="狼队白天抗推预言家的票型策略",
            summary="白天全力归票预言家形成票数优势，狼队利用预言家抗推制造混乱。",
            relevance=0.7,
        )
        # Near-duplicate: very high token overlap with hit_a.
        hit_b = _make_hit(
            entry_id="case_b",
            title="狼队白天抗推预言家票型策略与执行",
            summary="白天全力归票预言家形成票数优势，狼队利用预言家抗推执行制造混乱。",
            relevance=0.9,  # higher relevance, should win
        )
        # A clearly distinct hit (low Jaccard with the above two).
        hit_c = _make_hit(
            entry_id="case_c",
            title="女巫首夜解药保护预言家",
            summary="女巫首夜使用解药保护关键神牌，避免狼队夜里刀掉预言家。",
            relevance=0.6,
        )

        deduped = dedup_hits_by_similarity(
            [hit_a, hit_b, hit_c],
            max_items=2,
            similarity_threshold=0.6,
        )
        # hit_b wins (higher relevance), hit_c is distinct, hit_a dropped.
        ids = [h.entry_id for h in deduped]
        assert "case_b" in ids
        assert "case_a" not in ids
        assert "case_c" in ids
        assert len(deduped) == 2

    def test_distinct_hits_preserved(self) -> None:
        """When no two hits exceed the similarity threshold, all are kept
        (capped at max_items). The caller is responsible for relevance
        ordering — the helper preserves input order."""
        # Distinct titles + summaries with zero token overlap.
        hit_a = _make_hit(
            entry_id="a",
            title="狼推",
            summary="白天票",
            relevance=0.5,
        )
        hit_b = _make_hit(
            entry_id="b",
            title="女巫",
            summary="夜里药",
            relevance=0.6,
        )
        hit_c = _make_hit(
            entry_id="c",
            title="猎人",
            summary="开枪带",
            relevance=0.7,
        )
        # Pass them in relevance-descending order (caller's job).
        deduped = dedup_hits_by_similarity(
            [hit_c, hit_b, hit_a],
            max_items=2,
        )
        assert len(deduped) == 2
        # Highest-relevance hits first.
        assert deduped[0].entry_id == "c"
        assert deduped[1].entry_id == "b"

    def test_dedup_caps_at_max_items(self) -> None:
        """Dedup never returns more than ``max_items`` hits."""
        # Each hit's title+summary uses a disjoint single token so the
        # Jaccard similarity is exactly 0 (well below threshold).
        hits = [
            _make_hit(
                entry_id=f"x{i}",
                title=f"甲{i}",
                summary=f"乙{i}",
                relevance=0.5 - i * 0.01,
            )
            for i in range(5)
        ]
        deduped = dedup_hits_by_similarity(hits, max_items=2)
        assert len(deduped) == 2

    def test_dedup_caps_at_default(self) -> None:
        """G-R4-09: the cap is now caller-controlled. The module default
        (``_DEDUP_DEFAULT_MAX_ITEMS=2``) only applies when the caller
        does not pass ``max_items`` at all. Passing ``max_items=5``
        with 5 distinct hits returns 5 — the previous "module default
        as ceiling" behavior is removed.

        Sanity: calling dedup with no ``max_items`` (i.e. relying on
        the default) caps at 2.
        """
        from werewolf_agent.rag.prompt_renderer import (
            _DEDUP_DEFAULT_MAX_ITEMS,
        )

        hits = [
            _make_hit(
                entry_id=f"d{i}",
                title=f"甲{i}",
                summary=f"乙{i}",
                relevance=0.5 - i * 0.01,
            )
            for i in range(5)
        ]
        # Caller controls the cap.
        deduped = dedup_hits_by_similarity(hits, max_items=5)
        assert len(deduped) == 5, (
            f"G-R4-09: caller asked for 5; got {len(deduped)}. The "
            f"module default must NOT silently cap caller-supplied values."
        )
        # The module default still applies when no max_items is passed.
        deduped_default = dedup_hits_by_similarity(hits)
        assert len(deduped_default) == _DEDUP_DEFAULT_MAX_ITEMS

    def test_dedup_empty_input(self) -> None:
        assert dedup_hits_by_similarity([], max_items=2) == []

    def test_dedup_single_hit(self) -> None:
        hit = _make_hit(entry_id="solo")
        assert dedup_hits_by_similarity([hit], max_items=2) == [hit]

    def test_dedup_honors_caller_max_items(self) -> None:
        """G-R4-09: the module default (``_DEDUP_DEFAULT_MAX_ITEMS=2``)
        previously acted as an implicit ceiling — a caller asking for 3
        distinct hits silently got 2. The fix lets the caller control
        the cap: ``max_items=3`` with 3 distinct hits returns 3.

        The ``_DEDUP_DEFAULT_MAX_ITEMS=2`` default still applies when
        the caller does not pass ``max_items`` at all (so existing
        callers that relied on the old default keep working). The
        fix is that the default is no longer a CEILING — it is only
        a default for the argument.
        """
        # 3 distinct hits (zero token overlap, Jaccard ~ 0).
        hits = [
            _make_hit(
                entry_id=f"x{i}",
                title=f"甲{i}",
                summary=f"乙{i}",
                relevance=0.5 - i * 0.01,
            )
            for i in range(3)
        ]
        deduped = dedup_hits_by_similarity(hits, max_items=3)
        assert len(deduped) == 3, (
            f"G-R4-09: caller asked for 3 distinct hits; got {len(deduped)}. "
            f"The module default must NOT act as a ceiling."
        )
        # And the order matches the caller's input.
        assert [h.entry_id for h in deduped] == ["x0", "x1", "x2"]


# ===================================================================
# rag-hardening-1: renderer defense-in-depth
# ===================================================================


class TestPromptRenderHardening1:
    """rag-hardening-1: hits with ``allowed_in_live_context=False``
    must NEVER reach the live prompt, even if the retriever
    regresses and forgets to filter them out upstream. The slim
    renderer is the second line of defense and is responsible for
    dropping disallowed hits before rendering.
    """

    def test_hits_to_prompt_lines_drops_disallowed_hits(self) -> None:
        hit_allowed = _make_hit(entry_id="allowed_1", title="允许的")
        # ``RAGHit.allowed_in_live_context`` is auto-set by the
        # ``model_validator`` based on ``visibility`` — no need to
        # pass it explicitly.
        hit_disallowed = _make_hit(
            entry_id="forbidden_1",
            title="GOD_VIEW 内容",
            visibility=VisibilityBoundary.GOD_VIEW,
        )
        lines = hits_to_prompt_lines([hit_allowed, hit_disallowed])
        ids = [line["title"] for line in lines]
        assert "允许的" in ids
        assert "GOD_VIEW 内容" not in ids

    def test_render_hit_for_prompt_does_not_filter_itself(self) -> None:
        """The single-hit ``render_hit_for_prompt`` keeps its pure
        field-stripping contract — defense-in-depth lives in
        ``hits_to_prompt_lines`` (the live-prompt entry point), not
        in the per-hit pure function. A direct caller using
        ``render_hit_for_prompt`` with a disallowed hit would
        surface the hit; this is intentional so audit / moderator
        paths can still render individual hits for review.
        """
        hit = _make_hit(
            entry_id="forbidden",
            visibility=VisibilityBoundary.GOD_VIEW,
        )
        line = render_hit_for_prompt(hit)
        # No filter at this layer — pure field stripper.
        assert line["title"] == hit.title

    def test_hits_to_prompt_lines_allows_when_explicitly_allowed(self) -> None:
        """Hits with ``PLAYER_PERSPECTIVE`` (or any non-GOD/MOD
        visibility) auto-set ``allowed_in_live_context=True`` via
        the model validator and pass the renderer.
        """
        hit = _make_hit(
            entry_id="ok",
            visibility=VisibilityBoundary.PLAYER_PERSPECTIVE,
        )
        lines = hits_to_prompt_lines([hit])
        assert len(lines) == 1
        assert lines[0]["title"] == hit.title


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
