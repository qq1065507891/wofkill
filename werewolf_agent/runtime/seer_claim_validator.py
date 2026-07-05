# -*- coding: utf-8 -*-
"""Validate that public seer claims respect the 1-check-per-night rule.
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-05
    使用示例: 内部模块，无对外接口
In g_3528592081, fake Seer p08 publicly claimed "I checked p04 and p09 last
night (N1)" — but the Seer rule allows only 1 check per night. This module
detects such rule-violation leaks in public speeches.
Two helpers are exposed:
- :func:`extract_seer_claims` — regex extractor returning per-claim
``{night, target_id}`` records (night may be ``None`` if the speech
omits the night number).
- :func:`validate_seer_claim` — returns an error string when a speech
violates the 1-check-per-night rule or claims impossible future / pre-game
nights; returns ``None`` when the speech is clean.
"""

from __future__ import annotations

import re
from typing import Any

# Two patterns combined via alternation:
#   1) "我[在]?第? N 夜 查了 pNN ..."   — explicit-night branch
#   2) "(?:也|我) 查了 pNN"             — implicit-night branch (no night number)
_CLAIM_PATTERN = re.compile(
    r"我[在]?第?\s*(\d+)\s*夜[查验过]*了?\s*(?:是\s*)?(?:好人|狼人)?[，,。]?\s*(p\d{2})"
    r"|(?:也|我)\s*[查验过]*了?\s*(p\d{2})",
    re.IGNORECASE,
)


def extract_seer_claims(speech: str) -> list[dict[str, Any]]:
    """Extract seer-style claims from a public speech.

    Returns a list of dicts with keys: ``night`` (int | None),
    ``target_id`` (str). When the speech mentions a check without naming a
    night, ``night`` inherits the most recent explicit night earlier in
    the same speech (so "我第1夜查了p04，也查了p09" produces two claims
    both with ``night == 1``).
    """
    if not speech:
        return []
    claims: list[dict[str, Any]] = []
    last_explicit_night: int | None = None
    for m in _CLAIM_PATTERN.finditer(speech):
        night_str, target_with_night, target_no_night = m.groups()
        if target_with_night:
            night = int(night_str) if night_str else last_explicit_night
            if night is not None:
                last_explicit_night = night
            claims.append({"night": night, "target_id": target_with_night})
        elif target_no_night:
            claims.append({
                "night": last_explicit_night,
                "target_id": target_no_night,
            })
    return claims


def validate_seer_claim(speech: str, day_number: int) -> str | None:
    """Return an error message if the speech violates seer claim rules.

    Rules enforced:

    - At most 1 check claim per night.
    - Claims about future nights (``night > day_number``) are forbidden.
    - Claiming checks in night 0 (pre-game) is forbidden.

    Returns ``None`` when the speech contains no seer-style claims or
    when all claims are individually valid and respect the 1-per-night rule.
    """
    claims = extract_seer_claims(speech)
    if not claims:
        return None

    by_night: dict[int | None, list[str]] = {}
    for c in claims:
        by_night.setdefault(c["night"], []).append(c["target_id"])

    for night, targets in by_night.items():
        if night is not None and len(targets) > 1:
            return (
                f"公开发言中第{night}夜声称查验了 {len(targets)} 人，"
                f"违反预言家一夜只查一人的规则: {', '.join(targets)}"
            )

    for night in by_night:
        if night is not None and night > day_number:
            return (
                f"公开发言中声称第{night}夜查验，但当前是D{day_number}，"
                "未来夜晚的查验结果不可能已知"
            )

    if 0 in by_night:
        return "公开发言中声称第0夜查验，游戏开始前不可能有查验"

    return None
