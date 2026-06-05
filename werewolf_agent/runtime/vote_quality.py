"""Vote quality validation: require evidence-based voting.

Every vote must cite a concrete logic basis. Basis types:
seer_check, counterclaim, badge_flow, contradiction, vote_tally,
stance_reversal, pk_speech, speech_quote.
"""

from __future__ import annotations

import hashlib
import re
import random
from typing import Any
from enum import Enum

from werewolf_agent.core.models import GameState

VOTE_BASIS_VALUES = {
    "seer_check",
    "seer_siding",
    "speech_logic",
    "vote_pattern",
    "pressure_test",
    "anti_herd",
    "fallback",
}
SEER_STANCE_VALUES = {"trust", "distrust", "undecided", "no_claim"}
_UNEXPLAINED_VALUES = {"", "未说明", "无", "没有", "none", "null", "n/a"}

# Public alias used by other modules for retry hints (matches Pydantic enum).
VALID_VOTE_BASIS_VALUES = frozenset(VOTE_BASIS_VALUES)
VALID_SEER_STANCE_VALUES = frozenset(SEER_STANCE_VALUES)


# Basis detection patterns (Chinese)
_BASIS_PATTERNS: list[tuple[str, list[str]]] = [
    ("seer_check", [
        r"查杀", r"验[了过]?", r"查验", r"预言家.*?(?:查|验)", r"报.*?为",
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


def normalize_vote_basis(detected_bases: list[str]) -> str:
    """Map text evidence detectors to the public vote-basis enum.

    D-14: prefer the most informative basis when several are detected.
    Pre-fix the function returned on the *first* match in
    hard-coded priority order, which discarded multi-basis evidence
    (e.g., a vote that cited both a seer check AND a vote-tally
    pattern was reported as just ``seer_check`` even when the
    reasoning was anchored in the tally).  The fix ranks bases by
    evidentiary weight and picks the strongest.  Falls back to
    ``fallback`` when no basis is detected.
    """
    if not detected_bases:
        return "fallback"
    # Highest-evidence bases first.  A seer_check is the strongest
    # public signal; counterclaim / badge_flow rank as seer_siding
    # support; vote_tally is its own evidence class.
    _PRIORITY = (
        "seer_check",         # explicit wolf check from a seer
        "counterclaim",       # seer_siding: counterclaim was made
        "badge_flow",         # seer_siding: badge flow plan cited
        "contradiction",      # speech_logic: contradiction flagged
        "stance_reversal",    # speech_logic: stance change
        "speech_quote",       # speech_logic: prior speech quoted
        "vote_tally",         # vote_pattern: vote data cited
        "pk_speech",          # pressure_test: PK speech cited
    )
    for basis in _PRIORITY:
        if basis in detected_bases:
            # Map detector names to the public enum.
            if basis == "seer_check":
                return "seer_check"
            if basis in {"counterclaim", "badge_flow"}:
                return "seer_siding"
            if basis == "vote_tally":
                return "vote_pattern"
            if basis == "pk_speech":
                return "pressure_test"
            if basis in {"contradiction", "stance_reversal", "speech_quote"}:
                return "speech_logic"
    return "fallback"


def validate_structured_vote_action(
    action: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the full structured vote contract used in real games.

    Task 2: relax the basis regex — if no basis pattern is detected, default
    ``vote_basis`` to "fallback" rather than rejecting the vote. The strict
    regex caused 6/6 fallback votes in g_3528592081 because LLM retries could
    not satisfy the regex and the LLM-prompt cycle was wasted.
    """
    required_reason_fields = [
        ("reason", "投票理由"),
        ("suspect_reason", "怀疑理由"),
        ("not_voting_reason", "排除理由"),
        ("private_reason", "内心理由"),
    ]
    for field_name, label in required_reason_fields:
        value = str(action.get(field_name) or "").strip().lower()
        if value in _UNEXPLAINED_VALUES:
            return {
                "valid": False,
                "error_code": "vote_quality",
                "missing_field": field_name,
                "detected_bases": [],
                "hint": (
                    f"{label}不能为未说明；必须给出具体玩家、事件或逻辑依据。"
                    f"有效 vote_basis: {sorted(VALID_VOTE_BASIS_VALUES)}。"
                    f"有效 seer_stance: {sorted(VALID_SEER_STANCE_VALUES)}。"
                ),
            }

    # Normalize empty seer_stance to "no_claim" (the neutral default).
    raw_seer_stance = _string_value(action.get("seer_stance"))
    if not raw_seer_stance:
        action["seer_stance"] = "no_claim"
        seer_stance = "no_claim"
    else:
        seer_stance = raw_seer_stance
    if seer_stance not in SEER_STANCE_VALUES:
        return {
            "valid": False,
            "error_code": "vote_quality",
            "missing_field": "seer_stance",
            "detected_bases": [],
            "hint": (
                f"seer_stance必须是{SEER_STANCE_VALUES}之一。"
                f"有效 vote_basis: {sorted(VALID_VOTE_BASIS_VALUES)}。"
                f"有效 seer_stance: {sorted(VALID_SEER_STANCE_VALUES)}。"
            ),
        }

    # Normalize vote_basis: empty or out-of-enum values fall back to "fallback".
    raw_vote_basis = _string_value(action.get("vote_basis"))
    if not raw_vote_basis or raw_vote_basis not in VOTE_BASIS_VALUES:
        action["vote_basis"] = "fallback"
    else:
        action["vote_basis"] = raw_vote_basis

    # Task 2: relax the basis regex. If no basis pattern is detected in the
    # reason/speech text, accept the vote with vote_basis="fallback" rather
    # than rejecting. Empty votes that already passed the field checks above
    # are still valid here.
    reason_result = validate_vote_reason(action, context)
    detected_bases = reason_result.get("detected_bases", [])
    if not reason_result["valid"]:
        action["vote_basis"] = "fallback"
        if not _string_value(action.get("seer_stance")):
            action["seer_stance"] = "no_claim"

    return {
        "valid": True,
        "error_code": None,
        "missing_field": None,
        "detected_bases": detected_bases,
        "hint": "",
        "vote_basis": action.get("vote_basis"),
        "seer_stance": action.get("seer_stance"),
    }


def _string_value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value).strip()
    return str(value or "").strip()


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


def choose_vote_fallback_target(
    gs: GameState,
    voter_id: str,
    legal_targets: list[str],
) -> str | None:
    """Choose a deterministic fallback vote target from public evidence.

    This is used only after an agent failed to produce a valid vote. It avoids
    turning schema failures into a seat-order push by ranking legal targets by
    concrete current-day speech evidence, seer checks, and contradictions.
    """
    candidates = [target for target in legal_targets if target != voter_id]

    # Exclude wolf teammates when the voter is a werewolf
    voter = gs.players.get(voter_id)
    if voter and voter.role == "werewolf":
        candidates = [
            c for c in candidates
            if gs.players.get(c) is None or gs.players[c].role != "werewolf"
        ]

    if not candidates:
        return None

    scores = {target: 0 for target in candidates}

    # Speech evidence (same-day mentions with vote basis)
    for event in gs.events:
        if event.type != "speech" or event.payload.get("day_number") != gs.day_number:
            continue
        text = event.payload.get("text", "")
        bases = extract_vote_basis(text)
        if not bases:
            continue
        for target in candidates:
            if target in text:
                scores[target] += len(bases)

    # Public seer check claims: "wolf" alignment gets heavy weight
    # 仅使用公开的查杀声明，不直接读取 seer_check 私有事件
    try:
        from werewolf_agent.cognition.world_state import build_world_state
        _ws = build_world_state(gs)
        for f in _ws.facts_of_type("seer_check_claim"):
            val = (f.value or "").lower()
            target = f.target_player
            if target in candidates and ("wolf" in val or "狼" in (f.value or "")):
                scores[target] += 10
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "Failed to score vote targets with seer-check claims",
            exc_info=True,
        )

    # Contradiction alerts: players caught in contradictions get weight
    try:
        from werewolf_agent.cognition.world_state import build_world_state
        from werewolf_agent.cognition.contradiction import ContradictionEngine
        ws = build_world_state(gs)
        engine = ContradictionEngine()
        alerts = engine.detect(ws.facts, gs.day_number)
        for alert in alerts:
            if alert.player_id in candidates:
                scores[alert.player_id] += 3
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "Failed to score vote targets with contradiction alerts",
            exc_info=True,
        )

    best_score = max(scores.values(), default=0)
    if best_score > 0:
        best_targets = [target for target, score in scores.items() if score == best_score]
        return _stable_choice(gs, voter_id, best_targets)

    return _stable_choice(gs, voter_id, candidates)


def _stable_choice(gs: GameState, voter_id: str, candidates: list[str]) -> str:
    seed_parts = [gs.game_id, gs.day_number, voter_id, *candidates]
    raw = "|".join(str(part) for part in seed_parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0xFFFFFFFF
    return random.Random(seed).choice(list(candidates))
