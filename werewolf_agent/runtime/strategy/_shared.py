# -*- coding: utf-8 -*-
"""
功能描述：策略共享辅助函数——否定检测的单一来源（P-U2），供 wolf 和 hunter 模块共用。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
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
