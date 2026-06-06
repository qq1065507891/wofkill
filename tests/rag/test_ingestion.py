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
