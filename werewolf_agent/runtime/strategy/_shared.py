"""Shared strategy helpers.

P-U2: Single source of truth for negation detection used by both
``strategy.wolf`` and ``strategy.hunter``. Previously the two
modules each declared their own ``_NEGATION_WORDS`` / ``_NEGATION_RE``
/ ``_speech_is_negated``, with two subtle drift risks:
- The character classes were free to diverge silently.
- Each module's regex re-built the alternation at import time.

A single ``speech_is_negated(text) -> bool`` helper replaces all of
that. The context keywords are a superset of what wolf and hunter
needed individually, so the union does not regress either caller.
"""
from __future__ import annotations

import re
from typing import Final

# D-6: negation tokens that flip a power-role claim into a denial
# (e.g. "我不是预言家" / "我没查验" / "否认我是seer").  Wolf and
# hunter both use this set; keep additions in one place.
_NEGATION_WORDS: Final[frozenset[str]] = frozenset({
    "不是", "不", "没", "无", "非", "别", "未", "否认",
    "反", "否定",
    "绝对不是", "绝对不", "并没", "并非", "绝不",
})

# Power-role context keywords.  Union of:
#   wolf:   预言家 / seer / 查杀 / 金水 / 验了 / 查验
#   hunter: + 女巫 / 猎人 / witch / hunter
# A negation token is only "scoring" if one of these keywords is
# within 6 non-punctuation characters after it.
_POWER_ROLE_KEYWORDS: Final[str] = (
    r"预言家|女巫|猎人|seer|witch|hunter|查杀|金水|验了|查验"
)

_NEGATION_RE: Final[re.Pattern] = re.compile(
    r"(?:" + "|".join(re.escape(w) for w in _NEGATION_WORDS) + r")"
    r"[^，。,.\n]{0,6}"
    r"(?:" + _POWER_ROLE_KEYWORDS + r")",
    re.IGNORECASE,
)


def speech_is_negated(text: str) -> bool:
    """Return True if *text* contains a negation-of-power-role pattern.

    Catches phrases like ``"我不是预言家"``, ``"我没查验"``,
    ``"否认我是seer"``, ``"我不是女巫"``.  Used by both
    ``strategy.wolf`` (kill-target scorer + public-seer-claim
    detector) and ``strategy.hunter`` (shot-target scorer) so a
    denied claim does not trigger a "claimed_power_role" signal.
    """
    if not text:
        return False
    return bool(_NEGATION_RE.search(text))


__all__ = [
    "speech_is_negated",
    "_NEGATION_WORDS",
    "_NEGATION_RE",
]
