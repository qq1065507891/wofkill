"""Shared text-similarity helpers for evaluation and memory.

Same token regex as the original ``memory.reflection._token_set``:
ASCII word chars OR single CJK characters. Kept here (not in memory) so
``evaluation.attribution`` and ``memory.reflection`` share one source without
a brittle private-helper import. Verified import-cycle-free: ``evaluation``
imports nothing from ``memory.reflection``.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[一-鿿]")


def tokenize(text: str) -> set[str]:
    """Token set: lowercase ASCII runs + individual CJK characters."""
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def jaccard(left: str, right: str) -> float:
    """Jaccard similarity over token sets; 0.0 if either side is empty."""
    a = tokenize(left)
    b = tokenize(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
