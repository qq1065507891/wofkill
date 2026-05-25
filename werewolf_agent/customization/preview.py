"""Deterministic persona preview snippets."""

from __future__ import annotations

import html
from typing import Any


def build_persona_preview(persona: dict[str, Any]) -> dict[str, str]:
    """Return fixed preview utterances for the front-end persona panel.

    所有输出值经过 html.escape 转义，防止 XSS。
    """

    name = html.escape(str(persona.get("name") or f"Seat {persona.get('seat', '?')}"))
    style = html.escape(str(persona.get("speech_style") or "calm"))
    archetype = html.escape(str(persona.get("archetype") or "player"))
    logic_focus = html.escape(str(persona.get("logic_focus") or "medium"))
    aggression = html.escape(str(persona.get("aggression") or "medium"))
    return {
        "villager_opening": f"{name}: I am opening as a {style} {archetype}; my first pass keeps logic at {logic_focus}.",
        "defense": f"{name}: My defense is consistent with my previous position, and I will answer pressure at {aggression} intensity.",
        "wolf_night": f"{name}: Tonight I prefer a target that improves team tempo without exposing the wolf line.",
        "seer_claim": f"{name}: If I claim seer, I will give a clear check path and leave room for counter-claims.",
    }
