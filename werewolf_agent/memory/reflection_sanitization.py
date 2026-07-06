# -*- coding: utf-8 -*-
"""
清理反思文本中的玩家 ID、分段条目和过长来源文本。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.memory.reflection_sanitization import _scrub_ids
    >>> _scrub_ids("p07")
"""

from __future__ import annotations

import re

_PLAYER_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:p\d{1,3}|player[_-]?\d{1,3}|agent[_-]?\d{1,3})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_LLM_TRUTH_TOKENS = ("实际", "真实身份", "底牌", "查验结果", "死亡原因")
# Accepts markdown bullets (-/?/*) and numeric markers (1./1?/1)) with
# required trailing whitespace (verified g_1600154180 LLM output always
# emits "1. " with a space; if a future model emits "1.text" the marker
# won't strip and the residual prefix is visible noise, not silent corruption).
_LEADING_ITEM_PREFIX_RE = re.compile(r"^\s*(?:[-•*]|\d+[.、)])\s+")
_SOURCE_TEXT_CAP = 800


def _iter_section_items(body: str, *, min_chars: int = 6):
    """Yield cleaned content items from a reflection-section body.

    Accepts markdown bullets (``-``/``•``/``*``), numeric markers
    (``1.``/``1、``/``1)`` with trailing space), or plain prose lines.
    Strips the leading marker. Skips blank lines, items shorter than
    ``min_chars``, **and short colon-terminated list preambles** like
    "本局做对的:" (section intros, not items — exactly 6 chars, so
    ``len>=min_chars`` alone does NOT catch them; peer-review B1).

    Why: real LLM reflection output uses numeric markers and prose
    paragraphs, not markdown bullets; the previous
    ``startswith(("-", "•", "*")`` gate dropped everything.
    """
    for raw in body.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        cleaned = _LEADING_ITEM_PREFIX_RE.sub("", stripped, count=1).strip()
        if len(cleaned) < min_chars:
            continue
        # B1 fix: drop short colon-terminated list preambles
        # ("本局做对的:" / "本局做错的:" / "下局改进:") — section intros,
        # not items. Length guard (<=8) keeps real prose intact. Only
        # colon terminators (full/half-width): "。" is a normal prose
        # ender, not a preamble marker — including it would drop legit
        # short prose sentences (e.g. "投票站错边了。").
        if cleaned.endswith(("：", ":")) and len(cleaned) <= 8:
            continue
        yield cleaned


def _scrub_ids(text: str) -> str:
    return _PLAYER_ID_RE.sub("[玩家ID已省略]", str(text or ""))


def _cap_source_text(text: str, max_chars: int = _SOURCE_TEXT_CAP) -> str:
    cleaned = _scrub_ids(text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 6] + "…已截断"
