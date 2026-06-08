"""RAG seed ingestion validation tests.

Covers the boundary between the RAG knowledge base and base rules:
seeds must not contradict or duplicate rules that already live in
the system prompt, rule engine, or design document. Each test pins
down a specific seed (or seed family) to its fixed contract so
future edits do not silently reintroduce rule-truth or duplication.
"""

from __future__ import annotations

import re

import pytest

from werewolf_agent.rag.ingestion import create_seed_entries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed(entry_id: str):
    """Return the seed entry with ``entry_id`` (or fail loudly)."""
    entries = create_seed_entries()
    match = next((e for e in entries if e.entry_id == entry_id), None)
    if match is None:
        pytest.fail(f"seed entry {entry_id!r} not found in seed list")
    return match


def _seed_text(entry) -> str:
    """Concatenate the entry's user-visible text fields."""
    return " ".join(
        [
            entry.title,
            entry.summary,
            " ".join(entry.key_decisions),
        ]
    )


# ---------------------------------------------------------------------------
# G-R4-01 (P0): peace_night seed must include BOTH possible explanations
# ---------------------------------------------------------------------------


# Regex that flags "no-kill possibility presented only as a negation".
# The bug we want to catch is text that says "not wolf no-kill" (e.g.
# "不是狼人没刀", "不代表狼人没行动") — a negation that primes the
# LLM to dismiss the no-kill scenario entirely. The fix is to state
# the no-kill possibility positively (or as one of several options),
# so the regex matches any sentence-fragment that contains "no-kill
# wording preceded by a negation".
_NEGATED_NO_KILL_PATTERNS: tuple[str, ...] = (
    r"不是.{0,4}狼.{0,4}没刀",
    r"不是.{0,4}狼.{0,4}空刀",
    r"不代表.{0,4}狼.{0,4}没行动",
    r"不代表.{0,4}狼.{0,4}不行动",
    r"不代表.{0,4}狼.{0,4}没有行动",
)


def test_seed_peace_night_includes_no_kill_option() -> None:
    """G-R4-01: the seed for the peace-night case must NOT assert
    that "peace night = witch saved" as the only possibility. The
    base rule is that a peace night can also be the result of the
    wolves choosing not to kill (no-kill / 空刀). A seed that
    excludes that option misleads the LLM into thinking every
    peace night proves a witch save.

    The test scans title + summary + key_decisions and asserts
    that the no-kill possibility is stated positively (not as a
    negation that primes the LLM to dismiss it), and that the
    witch-antidote explanation is also present.
    """
    entry = _seed("seed_foundation_peace_night")
    text = _seed_text(entry)

    # Must mention the "witch used antidote / 银水" possibility.
    assert (
        "女巫" in text and ("银水" in text or "解药" in text)
    ), (
        "G-R4-01: peace-night seed dropped the witch-antidote "
        "explanation; both possibilities must be mentioned"
    )

    # Must mention the "wolf did not kill / 空刀" possibility.
    # We accept a positive statement; the negated form is
    # explicitly forbidden by the next assertion.
    assert ("没刀" in text) or ("空刀" in text) or ("不刀" in text) or (
        "没有选择击杀" in text
    ) or ("没有击杀" in text) or ("不选择击杀" in text) or (
        "狼.{0,4}选择.{0,4}不" in text
    ), (
        "G-R4-01: peace-night seed asserts 'witch saved' as the "
        "only explanation; it must also acknowledge that the "
        "wolves may have chosen not to kill"
    )

    # The title / summary / key_decisions must NOT contain a
    # negation that dismisses the no-kill possibility. The pre-fix
    # title literally said "不是狼人没刀" — which reads as
    # "NOT wolf no-kill" and primes the LLM to drop the scenario.
    for pattern in _NEGATED_NO_KILL_PATTERNS:
        assert not re.search(pattern, text), (
            f"G-R4-01: peace-night seed uses a negation that "
            f"dismisses the wolf no-kill possibility; pattern="
            f"{pattern!r}; text={text[:80]!r}"
        )


# ---------------------------------------------------------------------------
# G-R4-06 (P1): "基础常识" seeds must not duplicate base rules
# ---------------------------------------------------------------------------


# Phrases that name a numeric rule mechanic (1.5 vote weight) or a
# game-mechanic fact that already lives in the system prompt / design
# doc. The set is intentionally narrow so we only catch true
# duplicates, not generic community-vocabulary seeds (e.g. 金水).
_BASE_RULE_PHRASES: tuple[tuple[str, str], ...] = (
    # Sheriff 1.5x vote weight — lives in design doc Ch. 3 and
    # the system prompt. Mentioning it in a RAG seed is a
    # token-cost / confusion risk.
    (r"警.{0,4}1\.5.{0,4}票", "sheriff_vote_weight"),
    # Silver-water "saved by witch" wording that mirrors the
    # system-prompt base rule (a player who was saved by the
    # witch's antidote is publicly known to have been a wolf
    # target, but their faction is not publicly known).
    (r"银水.{0,8}被救", "silver_water_saved"),
)


def test_no_seed_duplicates_base_rules() -> None:
    """G-R4-06: ``基础常识`` seeds whose text repeats a base rule
    already in the system prompt / design doc are removed or
    relabelled as ``case_type=SPEECH_TEMPLATE``. The audit
    contract is "no RAG entry may restate a deterministic base
    rule" — these particular phrases encode numeric / mechanic
    facts that belong to the deterministic rules layer, not the
    RAG strategy layer.

    The test iterates every seed (not only the ``基础常识``
    family) so a future drift in case_type labelling is still
    caught.
    """
    entries = create_seed_entries()
    for entry in entries:
        text = _seed_text(entry)
        for pattern, label in _BASE_RULE_PHRASES:
            if re.search(pattern, text):
                pytest.fail(
                    f"G-R4-06: seed '{entry.entry_id}' duplicates base "
                    f"rule ({label}); pattern={pattern!r}; text snippet="
                    f"{text[:80]!r}"
                )


# ---------------------------------------------------------------------------
# rag-hardening-2 + rag-hardening-3: PII + identity-leak patterns
# ---------------------------------------------------------------------------


from werewolf_agent.rag.ingestion import CaseIngester, IngestionError
from werewolf_agent.rag.schemas import (
    CaseMetadata,
    QualityGrade,
    RAGEntry,
    ReviewStatus,
    SourceMetadata,
    SourceType,
    VisibilityBoundary,
    CaseType,
)


def _make_entry(
    *,
    entry_id: str = "rag_test_1",
    title: str = "测试条目",
    summary: str = "测试摘要",
    key_decisions: list[str] | None = None,
    tags: list[str] | None = None,
) -> RAGEntry:
    """Build a minimal RAGEntry for ingestion tests.

    NOTE: RAGEntry has a top-level ``tags`` field that is dropped
    silently by pydantic (the schema has no such attribute).
    The actual scanned field is ``metadata.tags``, so this helper
    maps the kwarg to ``metadata.tags`` to match the validator.
    """
    return RAGEntry(
        entry_id=entry_id,
        title=title,
        summary=summary,
        key_decisions=key_decisions or ["决策1", "决策2"],
        metadata=CaseMetadata(
            case_type=CaseType.ROLE_STRATEGY,
            quality_grade=QualityGrade.EXPERT_REVIEW,
            review_status=ReviewStatus.APPROVED,
            ruleset_id="pre_witch_hunter_idiot_mixed",
            player_count=12,
            phase="speech",
            role_perspective="any",
            visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
            tags=tags or ["test"],
            source=SourceMetadata(
                source_type=SourceType.EXPERT_COMMENTARY,
            ),
        ),
    )


class TestRagHardeningPII:
    """rag-hardening-2: PII / player-ID filter in
    ``_validate_forbidden_content``.

    The 12-player V1 board uses ``pNN`` exclusively, so the
    ``\bp\d{2}\b`` regex is a precise match. Seeds that name a
    specific player slot leak past-game identity that could match
    a real player in the current game.
    """

    def test_rejects_pnn_in_summary(self) -> None:
        ingester = CaseIngester()
        entry = _make_entry(summary="p05 是狼人,查杀了他")
        with pytest.raises(IngestionError, match="player-ID"):
            ingester.ingest(entry)

    def test_rejects_pnn_in_key_decisions(self) -> None:
        ingester = CaseIngester()
        entry = _make_entry(key_decisions=["p03 悍跳预言家失败"])
        with pytest.raises(IngestionError, match="player-ID"):
            ingester.ingest(entry)

    def test_rejects_pnn_in_tags(self) -> None:
        """rag-hardening-2: tags scanned too — a clean title/summary
        with a forbidden tag must still fail."""
        ingester = CaseIngester()
        entry = _make_entry(tags=["strategy", "p07"])
        with pytest.raises(IngestionError, match="player-ID"):
            ingester.ingest(entry)

    def test_accepts_generic_descriptions_without_pnn(self) -> None:
        ingester = CaseIngester()
        entry = _make_entry(
            title="基础常识：警上起跳",
            summary="预言家首夜空刀,悍跳狼失败",
        )
        # Should not raise — no p\d{2} token anywhere.
        ingester.ingest(entry)

    def test_pnn_regex_precise_to_two_digits(self) -> None:
        """The regex matches p01..p12 only, not p1 or p123.

        ``p1`` is one digit (not in V1), ``p123`` is three digits
        (also not in V1). Both should pass to avoid false
        positives against tokens like ``p3`` (page 3) or
        ``p1234`` (route id).
        """
        ingester = CaseIngester()
        # p1 (one digit) — should be allowed
        entry_short = _make_entry(entry_id="t1", summary="page p1 reference")
        ingester.ingest(entry_short)  # no raise
        # p12345 (5 digits) — should be allowed
        entry_long = _make_entry(entry_id="t2", summary="route p12345 reference")
        ingester.ingest(entry_long)  # no raise


class TestRagHardeningIdentityLeak:
    """rag-hardening-3: catch-all identity-leak patterns in
    ``_validate_not_rule_truth``. Reject generic
    ``pNN is <role>`` / ``pNN 查杀`` / ``pNN 金水`` assertions
    even when the player-ID filter doesn't fire (e.g. the
    wording uses "p03 是狼" without surrounding boundary
    that the PII regex still catches — defense in depth).
    """

    def test_rejects_pnn_is_werewolf_chinese(self) -> None:
        ingester = CaseIngester()
        # p\d{2} should be caught by PII filter; but also the
        # identity-leak patterns fire for phrasing like
        # "5号位是狼" (no p\d{2} token) — currently the
        # catch-all only covers pNN; non-pNN identity leaks
        # are out of scope.
        entry = _make_entry(summary="p05 是狼,查杀后好人崩盘")
        with pytest.raises(IngestionError, match="player-ID"):
            ingester.ingest(entry)

    def test_rejects_pnn_wolf_check_phrase(self) -> None:
        ingester = CaseIngester()
        entry = _make_entry(summary="p03 查杀 p05,预言家获胜")
        with pytest.raises(IngestionError, match="player-ID"):
            ingester.ingest(entry)


# ---------------------------------------------------------------------------
# rag-hardening-1 (seeds): peace_night must be split into
# any/public + werewolf-only entries
# ---------------------------------------------------------------------------


class TestSeedPeaceNightSplit:
    """rag-hardening-1: ``seed_foundation_peace_night`` (role_perspective=any)
    must NOT contain wolf-internal tactical framing; the wolf
    tactical content must live in a separate werewolf-only seed
    (``seed_foundation_peace_night_wolf``).
    """

    def test_public_peace_night_seed_is_any(self) -> None:
        """The public peace_night seed keeps role_perspective=any
        so good-side and wolf-side both retrieve it for the
        'peaceful night' situation, but the content must be
        framed from a good-side observation perspective — no
        explicit wolf tactical goals like '把解药骗掉'.
        """
        entry = _seed("seed_foundation_peace_night")
        meta = entry.metadata
        assert meta.role_perspective == "any", (
            f"public peace_night seed should stay role_perspective=any; "
            f"got {meta.role_perspective!r}"
        )

    def test_public_peace_night_seed_no_wolf_internal_framing(self) -> None:
        """The public seed must not contain wolf-internal tactical
        framing like '把解药骗掉' (trick the antidote) — those
        details belong in the werewolf-only companion seed.
        """
        entry = _seed("seed_foundation_peace_night")
        text = _seed_text(entry)
        # Wolf-internal tactical goals (NOT public-observable):
        forbidden_phrases = (
            "把解药骗掉",  # trick the antidote
            "拉长好人内讧",  # drag out good-side infighting
            "为毒药收割铺路",  # set up poison harvest
            "解药骗掉后再用毒药",  # older phrasing
            "狼队的合法策略",  # direct tactical framing
        )
        for phrase in forbidden_phrases:
            assert phrase not in text, (
                f"public peace_night seed contains wolf-internal "
                f"framing {phrase!r}; this leaks to villager LLM "
                f"agents and breaks role-perspective isolation. "
                f"Move wolf-internal content to seed_foundation_peace_night_wolf."
            )

    def test_wolf_peace_night_seed_exists_with_werewolf_perspective(self) -> None:
        """The wolf-only companion seed must exist with
        role_perspective=werewolf so villager LLM agents don't
        retrieve it (their retriever filter rejects werewolf
        perspective unless they ARE a wolf).
        """
        entry = _seed("seed_foundation_peace_night_wolf")
        meta = entry.metadata
        assert meta.role_perspective == "werewolf", (
            f"wolf peace_night seed should be werewolf-only; "
            f"got {meta.role_perspective!r}"
        )
        # Must mention the wolf tactical content we moved here
        text = _seed_text(entry)
        assert "骗" in text or "解药" in text, (
            "wolf peace_night seed should retain the wolf tactical content"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
