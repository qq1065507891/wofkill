"""Vote quality validation: require evidence-based voting.

Every vote must cite a concrete logic basis. Basis types:
seer_check, counterclaim, badge_flow, contradiction, vote_tally,
stance_reversal, pk_speech, speech_quote.
"""

from __future__ import annotations

import re
from typing import Any

from werewolf_agent.core.models import GameState


# Basis detection patterns (Chinese)
_BASIS_PATTERNS: list[tuple[str, list[str]]] = [
    ("seer_check", [
        r"查杀", r"验[了过]?", r"查验", r"预言家.*?(?:查|验)",
        r"(?:是|为)狼", r"金水",
    ]),
    ("counterclaim", [
        r"对跳", r"反跳", r"假预言家", r"真假预言家",
    ]),
    ("badge_flow", [
        r"警徽流", r"警徽", r"归票",
    ]),
    ("contradiction", [
        r"矛盾", r"前后不一", r"自相矛盾", r"逻辑不通",
        r"说[了过]?.*?但.*?又说",
    ]),
    ("vote_tally", [
        r"票数", r"投票.*?最高", r"上轮.*?票", r"得票",
    ]),
    ("stance_reversal", [
        r"立场.*?(?:反复|变化|转变)", r"翻供", r"改口",
        r"之前.*?现在.*?不一样",
    ]),
    ("pk_speech", [
        r"PK.*?(?:发言|说)", r"平票.*?(?:发言|说)",
    ]),
    ("speech_quote", [
        r"刚才说", r"之前说", r"上次说", r"说的是",
        r"原话", r"发言.*?(?:不合理|有问题)",
    ]),
]


def extract_vote_basis(reason: str) -> list[str]:
    """Extract logic basis types from a vote reason string.

    Returns list of basis type names that were detected.
    """
    if not reason or not reason.strip():
        return []

    found: list[str] = []
    for basis_name, patterns in _BASIS_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, reason):
                found.append(basis_name)
                break
    return found


def validate_vote_reason(
    action: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate that a vote action has at least one logic basis.

    Returns dict with:
    - valid: bool
    - missing_basis: bool (True when no basis found)
    - detected_bases: list[str]
    - hint: str (retry hint when invalid)
    """
    reason = action.get("reason", "")
    speech = action.get("speech", "")
    combined_text = f"{reason} {speech}"

    detected = extract_vote_basis(combined_text)

    if detected:
        return {
            "valid": True,
            "missing_basis": False,
            "detected_bases": detected,
            "hint": "",
        }

    return {
        "valid": False,
        "missing_basis": True,
        "detected_bases": [],
        "hint": (
            "投票理由缺少具体逻辑依据。请引用以下至少一种："
            "查验结果、对跳分析、警徽流、矛盾点、投票数据、"
            "立场变化、PK发言、或之前发言引用。"
        ),
    }


def build_day_discussion_summary(
    gs: GameState,
    day: int | None = None,
) -> list[dict[str, Any]]:
    """Build full discussion summary for a given day.

    Returns list of dicts with speaker, text, and metadata.
    """
    target_day = day if day is not None else gs.day_number
    speeches = []
    for event in gs.events:
        if event.type == "speech" and event.payload.get("day_number") == target_day:
            speeches.append({
                "speaker": event.payload.get("speaker", ""),
                "text": event.payload.get("text", ""),
                "day_number": target_day,
            })
    return speeches


def build_vote_pressure_context(
    gs: GameState,
    voter_id: str,
    pk_candidates: list[str] | None = None,
) -> dict[str, Any]:
    """Build vote pressure context for strategic voting.

    Includes PK candidates, prior vote results, and public pressure indicators.
    Wolf rush opportunity is informational only -- never forces a strategy.
    """
    context: dict[str, Any] = {
        "pk_candidates": pk_candidates or [],
        "prior_vote_results": [],
    }

    # Collect prior vote results
    for event in gs.events:
        if event.type == "vote_resolved":
            context["prior_vote_results"].append({
                "exiled": event.payload.get("exiled"),
                "reason": event.payload.get("reason"),
                "tied": event.payload.get("tied"),
            })

    # Wolf rush opportunity (informational)
    if pk_candidates and len(pk_candidates) == 2:
        context["rush_vote_opportunity"] = {
            "available": True,
            "description": "PK阶段可考虑冲票策略",
            "strategy_options": ["rush", "hook", "split", "abandon_teammate", "conservative"],
        }

    return context
