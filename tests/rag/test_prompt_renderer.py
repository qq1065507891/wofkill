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
        assert set(line.keys()) == {"title", "summary", "key_decisions"}

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
        hits = [
            _make_hit(entry_id=f"ext_{i:03d}", title=f"案例{i}")
            for i in range(5)
        ]
        lines = hits_to_prompt_lines(hits, max_items=2)
        assert len(lines) == 2
        assert lines[0]["title"] == "案例0"
        assert lines[1]["title"] == "案例1"

    def test_hits_to_prompt_lines_default_max_is_three(self) -> None:
        hits = [
            _make_hit(entry_id=f"ext_{i:03d}", title=f"案例{i}")
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
        assert decoded == [{"title": hit.title, "summary": hit.summary, "key_decisions": hit.key_decisions}]
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
